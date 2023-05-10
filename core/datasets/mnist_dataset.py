from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


DATA_DIR = "./data"


def get_mnist_dataloader(is_train: bool, batch_size: int, shuffle: bool) -> DataLoader:
    """Function for downloading the CIFAR 10 dataset and loading it into a DataLoader"""

    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=is_train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
