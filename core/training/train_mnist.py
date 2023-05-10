import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from typing import List, Tuple

from core.datasets.mnist_dataset import get_mnist_dataloader
from core.modules.forward_diffusion import Scheduler
from core.modules.loss_fn import LossFunction, ReverseDiffusionLoss
from core.modules.unet import UNet
from core.settings.logger import custom_logger
from core.settings.settings import get_config
from core.utils import SAVE_FOLDER


RESULTS_FOLDER = f"{SAVE_FOLDER}/training"


# Configs
config = get_config("MNIST_Training")
img_size = config["image_size"]
channels = config["channels"]


# Training loop
class DiffusionTrainer:
    """Class for handling the main operations while training a Diffusion Model"""

    def __init__(
        self,
        epochs: int,
        img_size: int,
        channels: int,
        save_and_sample_every: int = 1000,
    ) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using {device.upper()} as device!")
        self.device = torch.device(device)

        self.epochs = epochs
        self.img_size = img_size
        self.channels = channels
        self.save_and_sample_every = save_and_sample_every

    def train(
        self, model, dataloader: DataLoader, scheduler: Scheduler, loss_fn: LossFunction
    ):
        """Method for training the Diffusion Model"""

        model.to(self.device)
        diffusion_loss = ReverseDiffusionLoss(loss_fn)
        for epoch in range(self.epochs):
            for step, batch in enumerate(dataloader):
                # Sample t uniformally for every example in the batch
                batch_size = batch.shape[0]
                batch = batch.to(self.device)
                timesteps = torch.randint(
                    0, scheduler.max_timesteps, (batch_size,), device=self.device
                ).long()

                # Compute Loss
                optimizer.zero_grad()
                loss = diffusion_loss.get_loss(model, batch, timesteps)
                if step % 100 == 0:
                    print(
                        f"Loss at step {step} of epoch {epoch+1}: {round(loss.item(), 5)}"
                    )

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Save generated images
                if step != 0 and step % self.save_and_sample_every == 0:
                    milestone = step // self.save_and_sample_every
                    batches = self.images_per_batch(4, batch_size)
                    all_images_list = list(
                        map(
                            lambda n: self.p_sample_loop(
                                model,
                                batch_size=n,
                                max_timesteps=scheduler.max_timesteps,
                            ),
                            batches,
                        )
                    )
                    all_images = torch.cat(all_images_list, dim=0)
                    all_images = (all_images + 1) * 0.5
                    save_image(
                        all_images,
                        str(RESULTS_FOLDER / f"sample-{milestone}.png"),
                        nrow=6,
                    )

    @staticmethod
    def images_per_batch(num: int, divisor: int) -> List[int]:
        groups, remainder = divmod(num, divisor)
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    @torch.no_grad()
    def p_sample_loop(
        self, model, scheduler: Scheduler, batch_size: int, max_timesteps: int
    ):
        device = next(model.parameters()).device
        shape = (
            batch_size,
            self.channels,
            self.img_size,
            self.img_size,
        )

        # Start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, max_timesteps)),
            desc="sampling loop time step",
            total=max_timesteps,
        ):
            img = self.reverse_diffusion_step(
                model,
                scheduler,
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i,
            )
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def reverse_diffusion_step(
        model, scheduler: Scheduler, x: Tensor, t: Tensor, t_index: int
    ):
        betas_t = scheduler._extract_values(scheduler.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = scheduler._extract_values(
            scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = scheduler._extract_values(
            scheduler.sqrt_recip_alphas, t, x.shape
        )

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        posterior_variance_t = scheduler._extract_values(
            scheduler.posterior_variance, t, x.shape
        )
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Load data
dataloader = get_mnist_dataloader(**get_config("MNIST_Dataset"))

# Define UNet model
model = UNet(
    image_size=img_size,
    channels=channels,
    dim_mults=(1, 2, 4),
)

# Scheduler
max_timesteps = config["Scheduler"]
scheduler = Scheduler(max_timesteps)

# Optimizer and Loss
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = LossFunction.HUBER

# Trainer
trainer_configs = {
    "epochs": get_config("MNIST_Dataset")["batch_size"],
    "img_size": config["img_size"],
    "img_schannelsize": config["channels"],
    "save_and_sample_every": 1000,
}
trainer = DiffusionTrainer(**trainer_configs)

# Train model
trainer.train(model, dataloader, scheduler, loss_fn)
