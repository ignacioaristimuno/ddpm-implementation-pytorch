import torch
from torch import Tensor
import torch.nn.functional as F

from core.settings.logger import custom_logger
from core.utils import tensor_to_pillow


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'
    (Alex Nichol and Prafulla Dhariwal, 2021). https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int):
    """
    Linear schedule as proposed in original DDPM paper 'Denoising Diffusion Probabilistic
    Models' (Jonathan Ho et al., 2020). https://arxiv.org/pdf/2006.11239
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class Scheduler:
    """
    Custom class for hanlding the main operations of the Forward Diffusion Process.

    Formula:
        q(xt, x0) = N(xt; sqrt(āt)*x0, (1-āt)I)
    """

    def __init__(self, max_timesteps: int) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using {device.upper()} as device!")
        self.device = torch.device(device)

        self.max_timesteps = max_timesteps
        self.betas = linear_beta_schedule(timesteps=max_timesteps)
        alphas = 1 - self.betas

        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _extract_values(self, a: Tensor, t: Tensor, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def forward_diffusion(
        self, x_start: Tensor, t: Tensor, noise: Tensor = None
    ) -> Tensor:
        """Apply forward diffusion and return image as tensor (q sample)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract_values(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract_values(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_noisy_image(self, x_start: Tensor, t: Tensor) -> Tensor:
        """Apply forward diffusion and return image as PIL"""
        noisy_image = self.forward_diffusion(x_start, t=t)
        return tensor_to_pillow(noisy_image.squeeze())
