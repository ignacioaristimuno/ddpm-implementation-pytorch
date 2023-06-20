"""
Implementation of the loss function for a simple DDPM based on the
blog 'The Annotated Diffusion' by Hugging Face and adapted so the code is more
begginer-friendly for those looking to understande Diffusion Models.

https://huggingface.co/blog/annotated-diffusion
"""

from enum import Enum
from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F

from core.modules.forward_diffusion import Scheduler
from core.settings.settings import get_config
from core.settings.logger import custom_logger


logger = custom_logger("Loss Function")

scheduler = Scheduler(**get_config("Scheduler"))


class LossFunction(Enum):
    """Enum class for defining the available loss functions"""

    L1 = F.l1_loss
    MSE = F.mse_loss
    HUBER = F.smooth_l1_loss


class ReverseDiffusionLoss:
    """Class for handling the loss calculations for the reverse diffusion process"""

    def __init__(self, loss_fun: LossFunction) -> None:
        self.loss_fn = loss_fun

    def get_loss(self, denoise_model, x_start: Tensor, t: Tensor, noise: Tensor = None):
        """Method for calculating the loss for the predicted noise refered as p_losses"""

        if noise is None:
            noise = torch.rand_like(x_start)

        x_noisy = scheduler.forward_diffusion(x_start, t, noise)
        predicted_noise = denoise_model(x_noisy, t)
        return self.loss_fn(x_noisy, predicted_noise)
