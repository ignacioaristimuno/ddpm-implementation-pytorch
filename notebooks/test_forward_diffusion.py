import os
from PIL import Image
import torch
from typing import List

from core.modules.forward_diffusion import Scheduler
from core.settings.logger import custom_logger
from core.utils import (
    download_image,
    pillow_to_tensor,
    save_image,
    DUMMY_IMAGE_URL,
    SAVE_FOLDER,
)


logger = custom_logger("Testing Forward Diffusion")

scheduler = Scheduler(max_timesteps=999)


def test_forward_diffusion(timestep: int, url: str = None) -> Image:
    if not os.path.exists(f"{SAVE_FOLDER}/samples"):
        os.makedirs(f"{SAVE_FOLDER}/samples")

    image = download_image(url) if url else download_image()
    save_image(image.resize((128, 128)), "clean_image.png")
    x_start = pillow_to_tensor(image, image_size=128)

    timesteps = torch.tensor([timestep])
    noisy_images = scheduler.get_noisy_image(x_start, timesteps)
    save_image(noisy_images, f"noisy_image_{timestep}_steps.png")
    return noisy_images


timestep = 40
noisy_images = test_forward_diffusion(timestep, DUMMY_IMAGE_URL)
