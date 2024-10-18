# CIFAR10-Magnet-Loss
PyTorch implementation of the Magnet Loss, as proposed in the "Metric Learning with Adaptive Density Discrimination" paper, applied to the CIFAR-10 dataset.

**Running instructions**

1 - Pretrain the Inception network for three epochs on the ImageNet dataset (MiniImageNet from https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download-directory )

+ use [split_data.py](split_data.py)  to split th data in train and val and put it in the training and validation folder in a folder named ImageNet_2012
  ```
  python split_data.py
  ```
+ use [pretrain_imagenet.py](pretrain_imagenet.py) for pretraining
  ```
  python pretrain_imagenet.py
  ```
+ Alternatively you can download pretrained weights here:

2 - Use the pretraind model to train the the CIFAR10 data  with the magnet loss

+ use [train_magnet_new.py](train_magnet_new.py) for training.
  ```
  python train_magnet_new.py
  ```
+ You can adjust training parameters such as number of epochs, loss and the magnet loss parameters (K,M,D) in the file when loggong the parameters or pass then directly with the corresponding flags








