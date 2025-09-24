import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MLPBlock(nn.Module):
    """MLP block with time conditioning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First layer with time conditioning
        x = self.norm1(x)
        x = F.silu(self.linear1(x) + self.time_proj(time_emb))
        x = self.dropout(x)
        
        # Second layer
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x + residual


class Simple2DUNet(nn.Module):
    """
    Simple UNet-like architecture for 2D diffusion model.
    Takes 2D points and timestep, outputs noise prediction.
    """
    
    def __init__(
        self, 
        input_dim: int = 2,
        hidden_dims: Tuple[int, ...] = (128, 256, 512),
        time_embed_dim: int = 128,
        num_blocks_per_level: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            blocks = nn.ModuleList([
                MLPBlock(hidden_dim, hidden_dim * 2, time_embed_dim, dropout)
                for _ in range(num_blocks_per_level)
            ])
            self.encoder_blocks.append(blocks)
            
            if i < len(hidden_dims) - 1:
                self.encoder_downs.append(nn.Linear(hidden_dim, hidden_dims[i + 1]))
        
        # Middle block
        self.middle_block = MLPBlock(hidden_dims[-1], hidden_dims[-1] * 2, time_embed_dim, dropout)
        
        # Decoder (upsampling)
        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        print("Hidden dims:", hidden_dims)
        
        reversed_dims = list(reversed(hidden_dims))
        for i, hidden_dim in enumerate(reversed_dims[:-1]):
            next_dim = reversed_dims[i + 1]
            self.decoder_ups.append(nn.Linear(hidden_dim, next_dim))
            
            # After concatenation with skip connection, we have next_dim * 2
            # We need to project it back to next_dim for the residual connection
            concat_proj = nn.Linear(next_dim * 2, next_dim)
            blocks = nn.ModuleList([
                concat_proj,  # Project concatenated features back to next_dim
                *[MLPBlock(next_dim, next_dim * 2, time_embed_dim, dropout) 
                  for _ in range(num_blocks_per_level)]
            ])
            self.decoder_blocks.append(blocks)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims[0], input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 2)
            timestep: Timestep tensor of shape (batch_size,)
            
        Returns:
            Noise prediction of shape (batch_size, 2)
        """
        # Time embedding
        time_emb = self.time_embedding(timestep)
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder with skip connections
        skip_connections = []
        for blocks, down in zip(self.encoder_blocks[:-1], self.encoder_downs):
            for block in blocks:
                x = block(x, time_emb)
            skip_connections.append(x)
            x = F.silu(down(x))
        
        # Last encoder block (no downsampling)
        for block in self.encoder_blocks[-1]:
            x = block(x, time_emb)
        
        # Middle block
        x = self.middle_block(x, time_emb)

        # print("Middle block output shape:", x.shape)
        
        # Decoder with skip connections
        for up, blocks, skip in zip(self.decoder_ups, self.decoder_blocks, reversed(skip_connections)):
            x = F.silu(up(x))
            # print("Decoder upsample output shape:", x.shape)
            # print("Skip connection shape:", skip.shape)
            x = torch.cat([x, skip], dim=-1)  # Skip connection
            # print("After concatenation shape:", x.shape)
            
            # First block is the projection layer
            concat_proj = blocks[0]
            x = concat_proj(x)
            # print("After projection shape:", x.shape)
            
            # Apply remaining MLP blocks
            for block in blocks[1:]:
                x = block(x, time_emb)
            # print("Decoder block output shape:", x.shape)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class SimpleDiffusionModel(nn.Module):
    """
    Complete diffusion model with DDPM scheduler integration.
    """
    
    def __init__(
        self,
        unet: Optional[nn.Module] = None,
        input_dim: int = 2,
        **unet_kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        
        if unet is None:
            self.unet = Simple2DUNet(input_dim=input_dim, **unet_kwargs)
        else:
            self.unet = unet
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.unet(x, timestep)
    
    def training_step(self, batch: torch.Tensor, noise_scheduler) -> torch.Tensor:
        """
        Training step for diffusion model.
        
        Args:
            batch: Clean data of shape (batch_size, input_dim)
            noise_scheduler: DDPM scheduler from diffusers
            
        Returns:
            Loss tensor
        """
        batch_size = batch.shape[0]
        device = batch.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device, dtype=torch.long
        )
        
        # Sample noise
        noise = torch.randn_like(batch)
        
        # Add noise to clean images according to timestep
        noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)
        
        # Predict noise
        noise_pred = self(noisy_batch, timesteps)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int, 
        noise_scheduler,
        device: torch.device,
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate samples using DDPM sampling.
        
        Args:
            batch_size: Number of samples to generate
            noise_scheduler: DDPM scheduler from diffusers
            device: Device to run on
            num_inference_steps: Number of denoising steps
            
        Returns:
            Generated samples of shape (batch_size, input_dim)
        """
        # Start with random noise
        sample = torch.randn((batch_size, self.input_dim), device=device)
        
        # Set timesteps
        noise_scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self(sample, t.expand(batch_size).to(device))
            
            # Denoise
            sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
        
        return sample


# Example usage and factory functions
def create_simple_diffusion_model(
    input_dim: int = 2,
    hidden_dims: Tuple[int, ...] = (128, 256, 512),
    time_embed_dim: int = 128,
    num_blocks_per_level: int = 2,
    dropout: float = 0.1
) -> SimpleDiffusionModel:
    """Create a simple diffusion model for 2D data."""
    return SimpleDiffusionModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        time_embed_dim=time_embed_dim,
        num_blocks_per_level=num_blocks_per_level,
        dropout=dropout
    )


def create_lightweight_diffusion_model(input_dim: int = 2) -> SimpleDiffusionModel:
    """Create a lightweight diffusion model for quick experiments."""
    return SimpleDiffusionModel(
        input_dim=input_dim,
        hidden_dims=(64, 128),
        time_embed_dim=64,
        num_blocks_per_level=1,
        dropout=0.05
    )


# Example training setup
if __name__ == "__main__":
    # Example usage:
    from diffusers import DDPMScheduler
    import numpy as np
    from tqdm import tqdm
    from dataset import MultiModal2DDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    custom_matrix = np.array([
        [1],  # Mode 0 of x1 can pair with mode 0,1 of x2
        [1]
    ], dtype=bool)
    
    dataset_custom = MultiModal2DDataset(
        num_samples=50000,
        num_modes_x1=2,
        num_modes_x2=1,
        mode_spacing_x1=2.0,
        mode_spacing_x2=1.5,
        mode_std_x1=0.2,
        mode_std_x2=0.2,
        transition_matrix=custom_matrix
    )
    dataloader = DataLoader(dataset_custom, batch_size=256, shuffle=True)

    # Create model
    model = create_simple_diffusion_model()

    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    model.to(device)
    # Training loop (simplified)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 200

    for epoch in range(num_epochs):  # 5 epochs for example
        for batch in tqdm(dataloader, total=len(dataloader)):
            optimizer.zero_grad()
            batch = batch.to(device)
            loss = model.training_step(batch, scheduler)  # batch[0] contains the data points
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # save model
        if (epoch % 50 == 0 and epoch > 0) or epoch == num_epochs - 1:
            torch.save(model.state_dict(), "simple_diffusion_model_3.pth")

            # Sampling
            samples = model.sample(
                batch_size=1000,
                noise_scheduler=scheduler,
                device=device,
                num_inference_steps=50
            )

            print("Generated samples shape:", samples.shape)

            # ploy x1 at y= 0 and x2 at y=5, join with lines
            import matplotlib.pyplot as plt
            x1_samples = samples[:, 0].cpu().numpy()
            x2_samples = samples[:, 1].cpu().numpy()
            plt.figure(figsize=(8, 6))
            plt.scatter(x1_samples, np.zeros_like(x1_samples), color='blue', alpha=0.5, label='x1 samples (y=0)')
            plt.scatter(x2_samples, np.ones_like(x2_samples) * 5, color='red', alpha=0.5, label='x2 samples (y=5)')
            for i in range(len(x1_samples)):
                plt.plot([x1_samples[i], x2_samples[i]], [0, 5], color='gray', alpha=0.1)
            plt.title('Generated Samples from Diffusion Model')
            plt.xlabel('Value')
            plt.ylabel('Dummy Y-axis')
            plt.legend()
            plt.grid()
            plt.savefig(f"generated_samples_3_epoch_{epoch}.png")
