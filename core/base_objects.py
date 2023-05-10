from enum import Enum
from pydantic import BaseModel


class ImagesDataset(Enum):
    """Enum class for defining the allowed datasets to use"""

    MNIST: str = "MNIST"
    CELEB_A: str = "CELEB_A"


class DiffusionModelType(Enum):
    """Enum class for defining the allowed types of Diffusion Models"""

    MNIST_DDPM: str = "MnistDDPM"
