import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNetHorde(nn.Module):
    """
        This class implements the Horde model for the submitted strategy at CLVISION 2023
        Originally implemented by Benedikt Tscheschner, Marc Masana

        It mostly is the same as the slimmed ResNet by Lopez et al. in the GEM paper. Changes:
        - Added some functions to freeze modules
        - Remove the Linear head
    """
    def __init__(self, block, num_blocks, nf):
        super(ResNetHorde, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        # Track if the backbone is frozen
        self.frozen = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def freeze_backbone(self):
        """Freeze all parameters from the model"""
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        self.layer1.requires_grad_(False)
        self.layer2.requires_grad_(False)
        self.layer3.requires_grad_(False)
        self.layer4.requires_grad_(False)
        self.frozen = True

    def unfreeze_backbone(self):
        """Unfreeze all parameters from the model"""
        self.conv1.requires_grad_(True)
        self.bn1.requires_grad_(True)
        self.layer1.requires_grad_(True)
        self.layer2.requires_grad_(True)
        self.layer3.requires_grad_(True)
        self.layer4.requires_grad_(True)
        self.frozen = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        self.frozen = True

    def train(self, mode: bool = True):
        """Make sure that the BN are frozen when training"""
        super(ResNetHorde, self).train(mode)
        if self.frozen:
            self.freeze_bn()


def HordeSlimResNet18(n_classes, nf=20):
    """Slimmed ResNet18 adapted for Horde strategy"""
    return ResNetHorde(BasicBlock, [2, 2, 2, 2], nf)


__all__ = ["HordeSlimResNet18"]
