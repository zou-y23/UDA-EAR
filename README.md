# UDA-EAR
## Introduction
This repository provides the PyTorch implementation of UDA-EAR, an unsupervised domain adaptation method for egocentric action recognition. It uses a dual-branch pipeline with adaptive attention mechanisms to focus on key motion and interaction regions, combined with adversarial domain alignment to transfer fine-grained verb-noun knowledge across domains. 

## Framework
![]([https://github.com/zou-y23/UDA-EAR/blob/f14828aea76c51e9ceecf0d3f1c9fa3173a70005/framework.pdf](https://github.com/zou-y23/UDA-EAR/blob/226c957905514f17e0d55bb33735ec49cc5f0427/framework.png))

## Datasets
Prepare the datasets (EPIC-8 [1], and GTEA_KITCHEN-6 [2]) according to the instructions.

### EPIC-8
Download RGB frames from participants P01, P08 and P22 of the EPIC-KITCHENS-55 dataset, using official download [script](https://github.com/epic-kitchens/epic-kitchens-download-scripts). 

### GTEA_KITCHEN-6
Follow the instructions in [EgoAction](https://github.com/XianyuanLiu/EgoAction).

## References
[1] Munro, J. and Damen, D., "Multi-modal domain adaptation for fine-grained action recognition", in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp. 122-132.
[2] Liu, X., Zhou, S., Lei, T., Jiang, P., Chen, Z. and Lu, H., "First-person video domain adaptation with multi-scene cross-site datasets and attention-based methods", IEEE Transactions on Circuits and Systems for Video Technology, 2023, pp. 7774-7788.
