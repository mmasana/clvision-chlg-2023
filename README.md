# Horde @ CLVision Challenge 2023
This is the official submission repository from team _mmasana_ for the
[Continual Learning Challenge](https://sites.google.com/view/clvision2023/challenge)
held in the 4th [CLVision Workshop](https://sites.google.com/view/clvision2023/) @ CVPR 2023.

The proposed strategy to tackle the challenge scenarios is **Horde**, which has
been developed together as a team by Benedikt Tscheschner, Eduardo Veas and
Marc Masana.

## Horde description
The main idea of Horde is to tackle a few different issues from the challenge.
Here is how it works:
- we have some heuristics that decide at each experience if we want to learn a
feature extractor for the current classes, or instead just try to learn them
from the existing representations.
- when learning a new feature extractor, we train it with two heads that are
later discarded. The first one trains a typical CE-loss, while the other trains
a contrastive loss to promote the shape of the class representations to be
inside an n-dimensional ball. We allow a maximum of 10 feature extractors,
although none of the proposed scenarios need to reach that limit. The wrapper
for the different feature extractors does not seem to be larger than 1Gb (the
competition allows up to 4Gb). The CE-loss and the contrastive loss are balanced
with an adaptive strategy (`alpha` argument), which promotes that both losses
have a similar energy when backpropagated.
- regardless of a feature extractor being trained and added to the ensemble for
the current experience, we always train the unified head that takes all the
representations of each feature extractor and learns all seen classes.
- to balance that not all classes are seen at any given time, and since
rehearsal is not allowed, we keep track of mean and std of each class for each
feature extractor. Since all feature extractors are concatenated at their
output, we consider each mean and std per class to be the representations that
we store (hitting the competition maximum of 200, (mean+std)*100 classes).
- the method has some similarities to
[Fetril](https://openaccess.thecvf.com/content/WACV2023/papers/Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper.pdf)
(WACV, 2023), but allowing the learning of
multiple feature extractors instead of using a pre-trained backbone (not
allowed in this competition), and also adding the usage of the std and the
contrastive loss to improve the learned shape of the representations, and
thus avoiding the issues for the pseud-feature generation. The proposed
pseudo-feature generation strategy allows to apply our strategy to the
competition scenarios which contain class repetition.
- we have noticed that we can train each scenario under less than 150min
in our machines (competition restrictions are at 500min). Our proposed strategy
runs with `--num_epochs 20`, but we noticed that running more epochs was
usually beneficial. Therefore, if the organization sees fit, we would suggest
to run our submission both with `20` and with more epochs, if the time
restriction allows (e.g. `--num_epochs 50`).

## Added modifications
This repository implements the proposed Horde method by extending the provided
official [DevKit](https://github.com/ContinualAI/clvision-challenge-2023).

To make the evaluation of the solution easier, here is a brief description
of the main changes:
- **Model**: we remove the original linear head, and add some functions that allow
to freeze different parts of the model accordingly (i.e. backbone, BN layers).
The changes can be seen by checking the difference between the original
[`models/resnet_18.py`](./models/resnet_18.py) and our proposed
[`models/resnet_18_horde.py`](./models/resnet_18_horde.py).


- **Data augmentation**: we follow an established data augmentation technique for
this dataset type ([AutoAugment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf), CVPR 2019).
The transformations used are defined in
[`strategies/data_augmentation.py`](./strategies/data_augmentation.py),
and for simplicity, we replace the default transformations directly
in [`benchmarks/cir_benchmark.py`](benchmarks/cir_benchmark.py).


- **Horde**: our proposed strategy, which contains both the feature extractor's
model, and the two-phase strategy for training the feature extractors, and for
training the single unified head. The extended model, the training of the method,
and the definition of the contrastive loss pairs and losses are implemented in
[`strategies/horde.py`](./strategies/horde.py).


- **Logging**: a modification on the logging and verbosity of the training
process is implemented in [`utils/facil_logger.py`](./utils/facil_logger.py).
The main difference is just an adaptation on the metrics to report adapted
to our preferences.


- **Train/Main**: following the structure of the provided DevKit, the main
file is [`train.py`](./train.py), which has just been modified to call the
above-mentioned model, strategies and logger.


## Further questions
We have tried to make the code mostly self-explanatory and properly commented.
However, in case we have forgotten anything, please feel free to ask.
