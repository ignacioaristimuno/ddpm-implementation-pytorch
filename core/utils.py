from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from PIL import Image
import requests
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Resize,
    ToPILImage,
    ToTensor,
)
from typing import Union


SAVE_FOLDER = "data"
DUMMY_IMAGE_URL = "https://prod-images-static.radiopaedia.org/images/4170261/c5d7c3ed6c7fe53e59c2dd902e44b9_big_gallery.jpg"


def show_images(images: Union[ndarray, Tensor], title: str = "") -> None:
    """Shows a batch of images within a grid"""

    # Converting images to CPU numpy arrays
    if type(images) is Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(16, 16))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for _ in range(rows):
        for _ in range(cols):
            fig.add_subplot(rows, cols, idx + 1)
            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.show()


def show_images_first_batch(loader: DataLoader) -> None:
    """Shows the first batch of images within a torch DataLoader"""
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break


def pillow_to_tensor(image: Image, image_size: int) -> Tensor:
    """
    Function for getting the transform of an image as a PIL Image
    and returning a tensor of values between [-1, 1].
    """
    transform = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),  # Turn into Numpy array of shape HWC, divide by 255
            Lambda(lambda t: (t * 2) - 1),
        ]
    )
    image = transform(image)
    return image if len(image.shape) == 4 else image.unsqueeze(0)


def tensor_to_pillow(x: Tensor) -> Image:
    """
    Function for getting the reverse transform of an image as a tensor
    between [-1, 1] and returning the images as a PIL image.
    """
    if len(x.shape) == 4:
        x.squeeze()
    reverse_transform = Compose(
        [
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ]
    )
    return reverse_transform(x)


def download_image(url: str = DUMMY_IMAGE_URL) -> Image:
    """Download image from URL and return as"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def save_image(img: Image, img_name: str) -> None:
    """Function for saving an image in the specified path"""
    path = f"{SAVE_FOLDER}/samples/{img_name}"
    img.save(path)
