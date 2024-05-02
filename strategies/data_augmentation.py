from enum import Enum
import torchvision.transforms
from torchvision.transforms import AutoAugmentPolicy, Compose, Normalize,\
    RandomHorizontalFlip, RandomVerticalFlip, ToTensor, RandomCrop

from avalanche.benchmarks.utils import TupleTransform


class DataAugmentation(Enum):
    AutoAugment = "auto_augment"


def get_data_augmentation(augment_type: DataAugmentation):
    """
        This handles AutoAugment transforms, and can be easily extended with more options
        This is used together with the Horde strategy for the submitted strategy at CLVISION 2023
        Originally implemented by Benedikt Tscheschner, Marc Masana
    """
    if augment_type == DataAugmentation.AutoAugment:
        # The AutoAugment paper already checked that the CIFAR-10 transforms extend to CIFAR-100
        autoaugment = torchvision.transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
        image_transform = Compose([
                                   # Default CIFAR from the original dataset source
                                    RandomCrop((32, 32), padding=4, padding_mode="reflect"),
                                    RandomHorizontalFlip(),
                                    RandomVerticalFlip(),
                                    autoaugment,
                                    ToTensor(),
                                    Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])])
        target_transform = None
        return TupleTransform([image_transform, target_transform])
    else:
        raise RuntimeError("DataAugmentation Not Supported")
