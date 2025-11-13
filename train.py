import torch
from torch.utils.data import DataLoader

import numpy as np
import os
from tqdm import tqdm
from data import MultiModalDataset
from torch.utils.data import ConcatDataset

# Diffusers imports: prefer scheduler-specific import paths to satisfy static checkers
# DDPMScheduler is provided under diffusers.schedulers.scheduling_ddpm in some versions.
try:
    from diffusers import DDPMScheduler
except Exception:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from models import SimpleDiffusionModel
from utils import seed_everything

num_samples = 1000  # Adjust as needed
seed = 42  # For reproducibility
# Configure device between CUDA, MPS, and CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Set seed for reproducibility
seed_everything(42)

start_dataset = MultiModalDataset(
    num_samples=num_samples,
    start_means=[0.0],
    start_stds=[0.2],
    end_means=[0.75, -0.75],
    end_stds=[0.1, 0.1],
    transition_matrix=np.array([[1, 1]], dtype=bool),
    seed=seed,
)

bridge_dataset = MultiModalDataset(
    num_samples=num_samples * 4,
    start_means=[0.75, -0.75],
    start_stds=[0.1, 0.1],
    end_means=[0.75, -0.75],
    end_stds=[0.1, 0.1],
    transition_matrix=np.array([[1, 0], [0, 1]], dtype=bool),
    seed=seed,
)

end_dataset = MultiModalDataset(
    num_samples=num_samples,
    start_means=[0.75, -0.75],
    start_stds=[0.1, 0.1],
    end_means=[0.0],
    end_stds=[0.2],
    transition_matrix=np.array([[1], [1]], dtype=bool),
    seed=seed,
)

datasets = [start_dataset, bridge_dataset, end_dataset]
dataset = ConcatDataset(datasets)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

model = SimpleDiffusionModel().to(device)
scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 200
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("training", exist_ok=True)

print("Starting training loop on device: ", device)

accuracy = 0.0
pbar = tqdm(range(num_epochs), desc="Loss: 0.0")
for epoch in pbar:
    for batch in tqdm(dataloader, total=len(dataloader), leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        loss = model.training_step(
            batch, scheduler
        )  # batch[0] contains the data points
        loss.backward()
        optimizer.step()

    # save model
    if (epoch % 50 == 0 and epoch > 0) or epoch == num_epochs - 1:
        torch.save(model.state_dict(), "checkpoints/unified_diffusion_model.pth")

        # Sampling
        samples = model.sample(
            batch_size=1000,
            noise_scheduler=scheduler,
            device=device,
            num_inference_steps=50,
        )
        # accuracy = dataset.compute_accuracy(samples)
    pbar.set_description(f"Loss: {loss.item():.4f}, Acc: {accuracy:.3f}")
