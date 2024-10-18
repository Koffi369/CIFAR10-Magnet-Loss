from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch
import requests
import tarfile
import pickle


class magnet_CIFAR10(data.Dataset):
    """`CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where CIFAR-10 data is stored.
        train (bool, optional): If True, creates dataset from training data, otherwise from test data.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'cifar-10-train.pt'
    test_file = 'cifar-10-test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')

        if self.train:
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

        print("Size of dataset: " + str(len(self.data)))
        self.read_order = range(0, len(self.data))

    # def __getitem__(self, read_index):
    #     index = self.read_order[read_index]

    #     img, target = self.data[index], self.labels[index]

    #     # Convert to PIL Image
    #     img = Image.fromarray(img.numpy().transpose(1, 2, 0))  # Convert CHW to HWC

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target, index

    def __getitem__(self, read_index):
        index = self.read_order[read_index]
    
        img, target = self.data[index], self.labels[index]
    
        # Check the shape of the image before processing
        # print(f"Original image shape: {img.shape}")  # Debugging line
    
        # Reshape the flat image back to (3, 32, 32)
        img = img.view(3, 32, 32)  # Reshape from (3072,) to (3, 32, 32)
    
        # Convert to PIL Image
        img = img.numpy().transpose(1, 2, 0)  # Convert from CHW to HWC
    
        img = Image.fromarray(img.astype(np.uint8))  # Ensure the data type is uint8
    
        if self.transform is not None:
            img = self.transform(img)
    
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return img, target, index


    def __len__(self):
        return len(self.read_order)

    def update_read_order(self, new_order):
        self.read_order = new_order

    def default_read_order(self):
        self.read_order = range(0, len(self.data))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the CIFAR-10 data if it doesn't exist in processed_folder already."""
        if self._check_exists():
            return

        # Download the dataset
        print('Downloading ' + self.url)
        response = requests.get(self.url, stream=True)
        tar_path = os.path.join(self.root, 'cifar-10-python.tar.gz')
        with open(tar_path, 'wb') as f:
            f.write(response.content)

        # Extract the tar file
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=self.root)

        print('Processing...')

        # Load the data
        train_data, train_labels = self.load_cifar_batch(os.path.join(self.root, 'cifar-10-batches-py', 'data_batch_1'))
        for i in range(2, 6):
            batch_data, batch_labels = self.load_cifar_batch(os.path.join(self.root, 'cifar-10-batches-py', f'data_batch_{i}'))
            train_data = np.concatenate((train_data, batch_data))
            train_labels = np.concatenate((train_labels, batch_labels))

        test_data, test_labels = self.load_cifar_batch(os.path.join(self.root, 'cifar-10-batches-py', 'test_batch'))

        # Save processed data
        torch.save((torch.tensor(train_data), torch.tensor(train_labels)), os.path.join(self.root, self.processed_folder, self.training_file))
        torch.save((torch.tensor(test_data), torch.tensor(test_labels)), os.path.join(self.root, self.processed_folder, self.test_file))

        print('Done!')

    def load_cifar_batch(self, file):
        """Load a single batch of CIFAR-10."""
        with open(file, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            return batch['data'], np.array(batch['labels'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Transforms (if any): {}\n'.format(self.transform)
        fmt_str += '    Target Transforms (if any): {}'.format(self.target_transform)
        return fmt_str


    def get_classes(self, class_indices):
        """Return a subset of the dataset containing only the specified classes."""
        mask = np.isin(self.labels, class_indices)
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.read_order = range(len(self.data))  # Reset read order for new dataset
