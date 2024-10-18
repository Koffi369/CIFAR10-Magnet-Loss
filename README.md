# CIFAR10-Magnet-Loss
PyTorch implementation of the Magnet Loss, as proposed in the "Metric Learning with Adaptive Density Discrimination" paper, applied to the CIFAR-10 dataset.

**Running instructions**

1 - Pretrain the Inception network for three epochs on the ImageNet dataset (MiniImageNet from https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download-directory )


+ use [split_data.py](split_data.py)  to split th data in train and val and put it in the 
+ use [pretrain_imagenet.py](pretrain_imagenet.py) for pretraining







