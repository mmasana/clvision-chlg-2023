import copy
import numpy as np
from typing import List
from random import shuffle
from itertools import combinations

import torch
from torch.optim import Adam, Optimizer
from torch.nn import Linear, ModuleList, Module
from torch.utils.data import SubsetRandomSampler, DataLoader

from avalanche.training import Naive
from avalanche.evaluation.metrics import Accuracy, Mean, ClassAccuracy


def _set_learning_rate(optimizer: Optimizer, learning_rate):
    for g in optimizer.param_groups:
        g['lr'] = learning_rate


def pdist(vec):
    """
        Used to calculate the distances for the contrastive loss
        Adapted from https://github.com/adambielski/siamese-triplet
    """
    return -2 * vec.mm(torch.t(vec)) + vec.pow(2).sum(dim=1).view(1, -1) + vec.pow(2).sum(dim=1).view(-1, 1)


class HordeModel(torch.nn.Module):
    """
        This class implements the Horde model for the submitted strategy at CLVISION 2023
        Originally implemented by Benedikt Tscheschner, Marc Masana, Eduardo Veas
    """
    def __init__(self, num_feature_extractors, embedding_size):
        super(HordeModel, self).__init__()
        self.__num_classes = 100
        self.__num_fe = num_feature_extractors
        self.__embedding_size = embedding_size
        self.feature_extractors = ModuleList()
        self.linear = Linear(self.__embedding_size * self.__num_fe, self.__num_classes)

    def has_room_to_grow(self):
        return len(self.feature_extractors) < self.__num_fe

    def forward(self, x: torch.Tensor, return_features=False):
        """
            Applies the forward pass to the Horde model (all feature extractors, and the head)
            Args:
                x (tensor): input
                return_features (bool): return the embeddings before the head
        """
        features = torch.randn((x.size(0), self.__num_fe * self.__embedding_size), device=x.device)
        features = features * 0.01
        for m, model in enumerate(self.feature_extractors):
            features[:, m * self.__embedding_size: (m + 1) * self.__embedding_size] = model(x)
        if return_features:
            return self.linear(features), features
        else:
            return self.linear(features)

    def get_features(self, x, id_feature_extractor):
        """
            Applies the forward pass to a given feature extractor only
            Args:
                x (tensor): input
                id_feature_extractor (int): position of the feature extractor to evaluate
        """
        return self.feature_extractors[id_feature_extractor](x)

    def add_feature_extractor(self, feature_extractor: Module, position: int):
        feature_extractor.freeze_backbone()
        feature_extractor.eval()
        # Append FE
        if len(self.feature_extractors) < self.__num_fe:
            self.feature_extractors.append(feature_extractor)
        else:
            self.feature_extractors[position] = feature_extractor


# Pair Selectors for online contrastive loss calculations
class AllPositivePairSelector:
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    Adapted from https://github.com/adambielski/siamese-triplet
    """
    def __init__(self, balance=True):
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]]
        negative_pairs = all_pairs[labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]]
        if self.balance:
            # if more positive than negative pairs return all negatives that are available
            if len(positive_pairs) >= len(negative_pairs):
                return positive_pairs, negative_pairs
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector:
    """
    Creates all possible positive pairs. For negative pairs, pairs with the smallest distance are taken into
    consideration, matching the number of positive pairs.
    Adapted from https://github.com/adambielski/siamese-triplet
    """
    def __init__(self, cpu=True):
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]]
        negative_pairs = all_pairs[labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        # if more positive than negative pairs return all negatives that are available
        if len(positive_pairs) >= len(negative_distances):
            return positive_pairs, negative_pairs
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class HordeMLStrat(Naive):
    """
        Implements the Horde algorithm - Whaaarg!
        Originally implemented by Benedikt Tscheschner, Marc Masana, Eduardo Veas
    """

    def __init__(self, model, optimizer, number_feature_extractors: int, iterations_for_mean: int, ml_margin: float,
                 alpha: float, acc_thr: float, num_ml_dims: int, best_loss_model: bool, use_curr_cls_ph2: bool,
                 num_sim_feats: int, unk_mean: str, unk_std: float, **base_kwargs):
        self.__num_cls = 100
        self.__num_ml_dims = num_ml_dims
        self.__fe_out_sz = 160
        self.initial_model = model

        # Metric Learning args
        self.__pair_selector = HardNegativePairSelector()
        self.__margin = ml_margin
        self.__alpha = alpha

        # Best loss args
        self.__acc_thr = acc_thr
        self.__best_loss_model = best_loss_model

        # Horde args
        self.__num_fe = number_feature_extractors
        self.__num_iterations_for_mean = iterations_for_mean
        self.__use_curr_cls_ph2 = use_curr_cls_ph2
        self.__num_sim_feats = num_sim_feats
        self.__project_unknown_mean = unk_mean
        self.__project_unknown_std = unk_std

        # Build own Model with multiple feature extractors
        multi = HordeModel(number_feature_extractors, self.__fe_out_sz)
        super(HordeMLStrat, self).__init__(multi, optimizer, **base_kwargs)

        self.embedding_size = number_feature_extractors * self.__fe_out_sz
        self.std_embedding = torch.ones((self.__num_cls, self.__num_fe, self.__fe_out_sz), device=self.device)
        self.mean_embedding = torch.zeros((self.__num_cls, self.__num_fe, self.__fe_out_sz), device=self.device)

        self.max_samples_seen = torch.zeros((self.__num_cls, self.__num_fe))
        self.cls_trn_in_fe = torch.zeros((self.__num_cls, self.__num_fe))

        self.__clip_std_min = -2.0
        self.__clip_std_max = 2.0

        self.__initial_training_done = False
        self.mb_features = None

    def _get_cls_with_mean(self):
        return torch.clamp(self.max_samples_seen.sum(dim=1), 0, 1)

    def _get_cls_fe_trained(self):
        return torch.clamp(self.cls_trn_in_fe.sum(dim=1), 0, 1)

    def _get_cls_mean(self, feats, class_idx):
        cls_mean = 0.01 * torch.randn((self.__num_fe, self.__fe_out_sz)).to(self.device)
        for fe in range(len(self.model.feature_extractors)):
            if self.max_samples_seen[class_idx, fe] > 0:
                # When we have the mean calculated, we use it
                cls_mean[fe, :] = self.mean_embedding[class_idx, fe, :]
            else:
                if self.__project_unknown_mean == 'zeros':
                    cls_mean[fe, :] = 0.0
                elif self.__project_unknown_mean == 'feats':
                    cls_mean[fe, :] = feats[fe * self.__fe_out_sz:(fe + 1) * self.__fe_out_sz]
                elif self.__project_unknown_mean == 'noise':
                    # When random we just leave the already initialized random noise
                    pass
                else:
                    raise RuntimeError("Invalid project unknown mean method.")
        return cls_mean.ravel()

    def _get_cls_std(self, class_idx):
        cls_std = self.__project_unknown_std * torch.ones((self.__num_fe, self.__fe_out_sz)).to(self.device)
        for fe in range(len(self.model.feature_extractors)):
            if self.max_samples_seen[class_idx, fe] > 0:
                # When we have the std calculated, we use it
                cls_std[fe, :] = self.std_embedding[class_idx, fe, :]
            # When the std has not been calculated, then set the base one
        return cls_std.ravel()

    def _calculate_embedding_statistics(self):
        trn_dataset = self.experience.dataset.with_transforms("train")
        val_to_idx_dict = self.experience.dataset.targets.val_to_idx

        with torch.inference_mode():
            for cls_idx in val_to_idx_dict:
                cls_sampler = SubsetRandomSampler(indices=val_to_idx_dict[cls_idx])
                cls_loader = DataLoader(trn_dataset, batch_size=self.eval_mb_size, shuffle=False, sampler=cls_sampler)
                fe_embed_dict = {}
                for i in range(self.__num_iterations_for_mean):
                    for image, target, task in cls_loader:
                        image = image.to(self.device)

                        for num_fe, feature_extractor_i in enumerate(self.model.feature_extractors):
                            # Skip embedding calculation when better estimate was already calculated
                            if self.max_samples_seen[cls_idx, num_fe] >= len(val_to_idx_dict[cls_idx]):
                                continue

                            # TODO: speed-up, consider to remove/slice/index from the `from_num_samples`
                            feature_extractor_i.eval()
                            features = feature_extractor_i(image)
                            if num_fe not in fe_embed_dict:
                                fe_embed_dict[num_fe] = features
                            else:
                                fe_embed_dict[num_fe] = torch.vstack([fe_embed_dict[num_fe], features])

                for num_fe in fe_embed_dict.keys():
                    self.mean_embedding[cls_idx, num_fe, :] = torch.mean(fe_embed_dict[num_fe], dim=0)
                    self.std_embedding[cls_idx, num_fe, :] = torch.std(fe_embed_dict[num_fe], dim=0)
                    self.max_samples_seen[cls_idx, num_fe] = len(val_to_idx_dict[cls_idx])

    def contrastive_loss(self, outputs, targets):
        # Get image pairs
        positive_pairs, negative_pairs = self.__pair_selector.get_pairs(outputs, targets)
        if outputs.is_cuda:
            positive_pairs = positive_pairs.to(outputs.device)
            negative_pairs = negative_pairs.to(outputs.device)
        # Make sure that the sets are not empty
        if positive_pairs.nelement() == 0:
            return torch.tensor(0.0, device=outputs.device)
        # Apply the contrastive loss with margin
        positive_loss = (outputs[positive_pairs[:, 0]] - outputs[positive_pairs[:, 1]]).pow(2).sum(1)
        dist = (outputs[negative_pairs[:, 0]] - outputs[negative_pairs[:, 1]]).pow(2).sum(1).sqrt()
        negative_loss = torch.nn.functional.relu(self.__margin - dist).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

    def _train_feature_extractor(self, from_feature=None):
        if from_feature is None:
            # Starting FE with random weights
            main_model = copy.deepcopy(self.initial_model)
        else:
            # Inherit weights from given FE
            main_model = copy.deepcopy(self.model.feature_extractors[from_feature])
            main_model.train()
            main_model.unfreeze_backbone()
        # Create a head for CE and a head for Metric Learning
        ce_head = Linear(self.__fe_out_sz, self.__num_cls)
        ml_head = Linear(self.__fe_out_sz, self.__num_ml_dims)
        main_model.to(self.device)
        ce_head.to(self.device)
        ml_head.to(self.device)

        # Train schedule for feature extractor
        number_classes_exp = len(self.experience.dataset.targets.uniques)
        if number_classes_exp < 10:
            lr_schedule = [[0.001, 40], [0.0005, 20], [0.0001, 10]]
        else:
            lr_schedule = [[0.001, 70], [0.0005, 60], [0.0001, 50], [0.00005, 20]]

        print("Learning Feature Extractor... ")
        # LR will be overwritten later -- add params from main_model, CE head and ML head
        params = list(main_model.parameters()) + list(ce_head.parameters()) + list(ml_head.parameters())
        optimizer = Adam(params, lr=0.001)
        # Obtain Dataloader for current experience only
        loader = DataLoader(self.experience.dataset, batch_size=self.train_mb_size, shuffle=True)
        total_epochs = 0
        # Handle the combination of CE-loss and ML-loss
        if self.__alpha == -1.0:
            # Adaptive alpha: alpha promotes mostly CE-loss at the beginning
            wk_alpha = 0.001
        else:
            # The alpha value is fixed across all the training session
            wk_alpha = self.__alpha
        # Starting best model is current model
        best_loss = np.inf
        best_model = None
        if self.__best_loss_model:
            best_model = copy.deepcopy(main_model)
        # Training loop
        for schedule in lr_schedule:
            _set_learning_rate(optimizer, schedule[0])

            for epoch in range(schedule[1]):
                total_epochs += 1
                metric_ce_loss = Mean()
                metric_ml_loss = Mean()
                metric_acc = Mean()
                batch_acc = Accuracy()

                for x, y, _ in loader:
                    x_device = x.to(self.device)
                    y_device = y.to(self.device)
                    feats = main_model(x_device)
                    # Forward through the CE-head
                    ce_out = ce_head(feats)
                    ce_loss = torch.nn.functional.cross_entropy(ce_out, y_device)
                    metric_ce_loss.update(ce_loss.detach().item(), weight=y.size(0))
                    # Apply working alpha
                    losses = (1 - wk_alpha) * ce_loss
                    if wk_alpha > 0.0:
                        # Forward through the ML-head
                        ml_out = ml_head(feats)
                        ml_loss = self.contrastive_loss(ml_out, y_device)
                        losses += wk_alpha * ml_loss
                        metric_ml_loss.update(ml_loss.detach().item(), weight=y.size(0))
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    batch_acc.update(ce_out.argmax(dim=1), y_device)
                    metric_acc.update(batch_acc.result(), weight=y.size(0))
                    batch_acc.reset()
                print(f"FE Training {self.clock.train_exp_counter} | Epoch {total_epochs} "
                      f"| CE Loss {metric_ce_loss.result():.4f} | ML Loss {metric_ml_loss.result():.4f} "
                      f"| Train Acc {metric_acc.result() * 100.0:.2f} | Alpha {wk_alpha:.4f}")

                # Keep track of the model with the best (lowest) loss
                if self.__best_loss_model:
                    train_loss = (1 - wk_alpha) * metric_ce_loss.result() + wk_alpha * metric_ml_loss.result()
                    if train_loss < best_loss:
                        best_loss = train_loss
                        best_model = copy.deepcopy(main_model)

                # Adaptive alpha: balance the working alpha to accommodate the loss difference in magnitude
                if self.__alpha == -1.0:
                    wk_alpha = metric_ml_loss.result() / (metric_ml_loss.result() + metric_ce_loss.result() + 1e-12)

        # Recover the best model
        if self.__best_loss_model and best_model is not None:
            main_model = copy.deepcopy(best_model)
        # Return the trained feature extractor
        return main_model

    def __calculate_class_accuracy_scores(self):
        cur_train_dataset = self.experience.dataset.with_transforms("eval")
        dataloader = DataLoader(cur_train_dataset, batch_size=self.eval_mb_size)
        class_acc = ClassAccuracy()
        class_acc.reset()
        with torch.inference_mode():
            self.model.eval()
            for x, y, task in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                class_acc.update(y, pred, task_labels=0)
        self.model.train()
        return class_acc.result()[0]

    def _before_training_exp(self, **kwargs):
        super(HordeMLStrat, self)._before_training_exp(**kwargs)

        # Calculate FZ for current data -- get performance and make deepcopy of model
        self.__backup_model = copy.deepcopy(self.model)
        self.cls_sc_before = self.__calculate_class_accuracy_scores()

        # Decide if we want to build a feature extractor for this experience or not
        seen_cls_fe = self._get_cls_with_mean()
        trained_cls_fe = self._get_cls_fe_trained()
        new_untrained_fe_cls = [cls for cls in self.experience.dataset.targets.uniques if trained_cls_fe[cls] == 0]
        # Train new FE on first exp or when fewer than 85 classes have been trained, and 5 of those are completely new
        if not self.__initial_training_done or (seen_cls_fe.sum() < 85 and len(new_untrained_fe_cls) >= 5):
            # Check if the maximum amount of FE has been reached
            if self.model.has_room_to_grow():
                # If there is room for another FE, add it
                new_fe_pos = len(self.model.feature_extractors)
                feature_extractor = self._train_feature_extractor()
                self.model.add_feature_extractor(feature_extractor, new_fe_pos)
                self.cls_trn_in_fe[list(self.experience.dataset.targets.uniques), new_fe_pos] = 1
            else:
                # Figuring out which FE to replace. Currently just the one with the least amount of features.
                min_fe_cls, min_fe_pos = self.cls_trn_in_fe.sum(dim=0).min(dim=0)
                if len(self.experience.dataset.targets.uniques) > min_fe_cls:
                    # Replace the oldest FE with the least amount of classes
                    feature_extractor = self._train_feature_extractor(min_fe_pos)
                    # Updated which classes have been trained on the FE
                    self.model.add_feature_extractor(feature_extractor, min_fe_pos)
                    self.max_samples_seen[:, min_fe_pos] = 0
                    self.cls_trn_in_fe[:, min_fe_pos] = 0
                    self.cls_trn_in_fe[list(self.experience.dataset.targets.uniques), min_fe_pos] = 1
                else:
                    # If the new FE would have fewer classes than any existing one, then we ignore adding it
                    print("The new feature extractor was not good enough for this stack.")

        self._calculate_embedding_statistics()

    def forward(self):
        output, self.mb_features = self.model(self.mb_x, return_features=True)
        return output

    def _before_backward(self, **kwargs):
        super(HordeMLStrat, self)._before_backward(**kwargs)
        # Addition of the Horde loss (inspired by FeTrIL)
        horde_loss = torch.tensor(0.0).to(self.device)
        curr_exp_cls = self.experience.dataset.targets.uniques
        for i, y in enumerate(self.mb_y):
            pseudo_feats, pseudo_targets = [], []
            # Extract label and features from current sample
            orig_cls = y.cpu().item()
            orig_feats = self.mb_features[i].detach()
            # Choose new valid classes to simulate/hallucinate
            all_sim_cls = [idx for idx, valid in enumerate(self._get_cls_with_mean()) if valid and idx != orig_cls]
            # Remove the classes from the current experience
            if not self.__use_curr_cls_ph2:
                all_sim_cls = [elem for elem in all_sim_cls if elem not in curr_exp_cls]
            if len(all_sim_cls) > 0:
                # Randomly choose as many simulated classes as needed
                shuffle(all_sim_cls)
                all_sim_cls = all_sim_cls[:min(self.__num_sim_feats, len(all_sim_cls))]
                # Simulate each new feature
                for sim_cls in all_sim_cls:
                    # Calculate offset from original class
                    orig_cls_offset = orig_feats - self._get_cls_mean(orig_feats, orig_cls)  # remove orig mean
                    orig_cls_offset = orig_cls_offset / (self._get_cls_std(orig_cls) + 1e-8)  # project orig std
                    orig_cls_offset = torch.clamp(orig_cls_offset, self.__clip_std_min, self.__clip_std_max)
                    # project and shift with the simulated class mean and std
                    sim_feat = self._get_cls_mean(orig_feats, sim_cls) + orig_cls_offset * self._get_cls_std(sim_cls)
                    sim_feat = sim_feat.detach().unsqueeze_(0)
                    # Store additional class elements
                    pseudo_feats.append(sim_feat)
                    pseudo_targets.append(sim_cls)
                # Stack pseudo-features and pass them through the head
                pseudo_feats = torch.vstack(pseudo_feats)
                pseudo_targets = torch.tensor(pseudo_targets).to(self.device)
                output = self.model.linear(pseudo_feats)
                pseudo_loss = torch.nn.functional.cross_entropy(output, pseudo_targets)
                horde_loss += pseudo_loss
        # Normalize and add to the main training loss
        self.loss += horde_loss / len(self.mb_y)

    def _after_training(self, **kwargs):
        super(HordeMLStrat, self)._after_training(**kwargs)
        self.__initial_training_done = True
        # Calculate performance of current data
        self.cls_sc_after = self.__calculate_class_accuracy_scores()
        # If performance is not better than FZ from before training, recover old model
        mean_acc_before = sum(self.cls_sc_before.values()) / len(self.cls_sc_before.values())
        mean_acc_after = sum(self.cls_sc_after.values()) / len(self.cls_sc_after.values())
        if mean_acc_before >= self.__acc_thr * mean_acc_after:
            self.model = copy.deepcopy(self.__backup_model)
            print("I've just ignored what I trained!")
        # Delete old model
        self.__backup_model = None
