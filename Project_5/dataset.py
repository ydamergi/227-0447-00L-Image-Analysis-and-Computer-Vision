import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    A standard image dataset that can be used for training and testing.
    The images are converted to tensors and normalized and (H, W, C) -> (C, H, W)
    """

    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.299,0.224,0.225))
    ])
    
    def __init__(self, images, labels=None):
        """
        Initialize the dataset with the images and labels.
        
        Args:
          images: a numpy array of shape (N, H, W, C) containing the images
          labels: a numpy array of shape (N,) containing the labels
        """

        self.images = images
        if labels is None:
            labels = np.zeros(len(images))
        self.labels = labels
        self.unique_labels = list(set(labels))
        self.num_classes = len(self.unique_labels)

        image_size = images.shape[1:]
        self.image_size = image_size
        self.data_size = (image_size[2], image_size[0], image_size[1])
        
    def __len__(self):
        """ Return the number of images in the dataset """
        return len(self.images)

    def __getitem__(self, idx):
        """ Return the image and label at the given index """
        image = self.transform(self.images[idx])
        label = self.labels[idx]
        return image, label


def get_datasets_from_h5(h5_path):
    """
    Loads a dataset from an h5 file.
    """
    with h5py.File(h5_path, "r") as f:
        train_images = f["train_images"][:]
        train_labels = f["train_labels"][:]
        val_images = f["val_images"][:]
        val_labels = f["val_labels"][:]
        test_images = f["test_images"][:]
        test_labels = f["test_labels"][:]
    return (ImageDataset(train_images, train_labels),
        ImageDataset(val_images, val_labels),
        ImageDataset(test_images, test_labels)
    )


def get_test_loader(test_h5_path):
    with h5py.File(test_h5_path, "r") as f:
        images = f["images"][:]
    test_dataset = ImageDataset(images)
    return DataLoader(test_dataset, batch_size=50, shuffle=False)


def get_loaders_from_datasets(train_dataset, val_dataset, test_dataset, batch_size=50):
    """
    Takes the datasets and returns the train, validation, and test loaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader