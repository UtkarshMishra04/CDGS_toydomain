import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers import DDPMScheduler
from utils import load_models
from diffusers.utils.torch_utils import randn_tensor
from models import SimpleDiffusionModel


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_views(num_models):
    """
    Create sliding 2D views over the latent space
    For M models, views are (0,2), (1,3), ..., (M-1, M+1)
    Total latent dimension = M+1
    """
    views = [(i, i + 2) for i in range(num_models)]
    print(f"Created {len(views)} views: {views}")
    print(f"Total latent dimension: {views[-1][1]}")
    return views


def undo_step(latents, timestep, scheduler, generator=None):
    """
    Reverse a single diffusion denoising step to return to a noisier state.

    This function is used in the iterative resampling procedure to "undo" a
    denoising step, allowing the sampler to re-denoise from the same timestep
    multiple times for better compositional alignment.

    Args:
        latents: current latent state (B, dim)
        timestep: the current timestep in the scheduler
        scheduler: DDPMScheduler instance
        generator: optional random generator for reproducibility

    Returns:
        latents: noisier latent state (one step back in the forward process)
    """
    n = scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    for i in range(n):
        beta = scheduler.betas[timestep + i]
        if latents.device.type == "mps":
            # randn does not work reproducibly on mps
            noise = randn_tensor(
                latents.shape, dtype=latents.dtype, generator=generator
            )
            noise = noise.to(latents.device)
        else:
            noise = randn_tensor(
                latents.shape,
                generator=generator,
                device=latents.device,
                dtype=latents.dtype,
            )

        # Apply forward diffusion step: x_t = sqrt(1-beta) * x_{t-1} + sqrt(beta) * noise
        latents = (1 - beta) ** 0.5 * latents + beta**0.5 * noise

    return latents


def compute_inversion_scores_single(batched_x0s, views, model, scheduler, device):
    """
    Compute smoothness scores using a single model with batched inference.

    Args:
        batched_x0s: clean predictions for each view, shape [num_models, B, local_dim]
        views: list of (start, end) tuples for each local plan
        model: single model instance used for all views
        scheduler: DDPMScheduler instance
        device: torch device

    Returns:
        final_scores: smoothness scores for each sample, shape (B,)
    """
    num_models = len(views)
    B = batched_x0s.shape[1]

    all_timesteps = scheduler.timesteps.flip(dims=(0,))
    num_timesteps = len(all_timesteps)
    inversion_latents = batched_x0s.clone()
    all_noise_prediction = []

    inversion_steps = num_timesteps // 20
    for idx in tqdm(
        range(inversion_steps), leave=False, desc="Computing inversion scores"
    ):
        t = all_timesteps[idx]
        t_next = all_timesteps[idx + 1]

        alpha_t = scheduler.alphas_cumprod[t]
        alpha_t_next = scheduler.alphas_cumprod[t_next]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)

        with torch.no_grad():
            # Batch all views together: (num_models, B, local_dim) -> (num_models * B, local_dim)
            batched_input = inversion_latents.reshape(num_models * B, -1)
            batched_t = t.repeat(num_models * B).to(device)

            # Single forward pass for all views
            batched_noise_pred = model(batched_input, batched_t)
            noise_pred_combined = batched_noise_pred.reshape(num_models, B, -1)

            # DDIM inversion
            x0_pred = (
                inversion_latents - sqrt_one_minus_alpha_t * noise_pred_combined
            ) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            noise_pred_combined = (
                inversion_latents - sqrt_alpha_t * x0_pred
            ) / sqrt_one_minus_alpha_t
            inversion_latents = (
                sqrt_alpha_t_next * x0_pred
                + sqrt_one_minus_alpha_t_next * noise_pred_combined
            )

            all_noise_prediction.append(noise_pred_combined)

    all_intermediate_noise_preds = torch.stack(all_noise_prediction, dim=2)
    derivative = torch.diff(all_intermediate_noise_preds, dim=2)
    all_scores = torch.norm(derivative.reshape(num_models * B, -1), dim=1).reshape(
        num_models, B
    )
    final_scores = all_scores.mean(dim=0)

    return final_scores


def compute_inversion_scores_separate(batched_x0s, views, models, scheduler, device):
    """
    Compute smoothness scores using separate models for each view.

    Args:
        batched_x0s: clean predictions for each view, shape [num_models, B, local_dim]
        views: list of (start, end) tuples for each local plan
        models: list of model instances
        scheduler: DDPMScheduler instance
        device: torch device

    Returns:
        final_scores: smoothness scores for each sample, shape (B,)
    """
    num_models = len(views)
    B = batched_x0s.shape[1]

    all_timesteps = scheduler.timesteps.flip(dims=(0,))
    num_timesteps = len(all_timesteps)
    inversion_latents = batched_x0s.clone()
    all_noise_prediction = []

    inversion_steps = num_timesteps // 20
    for idx in tqdm(
        range(inversion_steps), leave=False, desc="Computing inversion scores"
    ):
        t = all_timesteps[idx]
        t_next = all_timesteps[idx + 1]

        alpha_t = scheduler.alphas_cumprod[t]
        alpha_t_next = scheduler.alphas_cumprod[t_next]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)

        with torch.no_grad():
            # Get noise predictions from each model separately
            noise_pred_combined = torch.zeros_like(inversion_latents)
            for id, (start, end) in enumerate(views):
                latent_view = inversion_latents[id]
                noise_pred = models[id](latent_view, t.repeat(B).to(device))
                noise_pred_combined[id] = noise_pred

            # DDIM inversion
            x0_pred = (
                inversion_latents - sqrt_one_minus_alpha_t * noise_pred_combined
            ) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            noise_pred_combined = (
                inversion_latents - sqrt_alpha_t * x0_pred
            ) / sqrt_one_minus_alpha_t
            inversion_latents = (
                sqrt_alpha_t_next * x0_pred
                + sqrt_one_minus_alpha_t_next * noise_pred_combined
            )

            all_noise_prediction.append(noise_pred_combined)

    all_intermediate_noise_preds = torch.stack(all_noise_prediction, dim=2)
    derivative = torch.diff(all_intermediate_noise_preds, dim=2)
    all_scores = torch.norm(derivative.reshape(num_models * B, -1), dim=1).reshape(
        num_models, B
    )
    final_scores = all_scores.mean(dim=0)

    return final_scores


def rearrange_batch_by_scores(latents, scores, top_K):
    """
    Rearrange batch by selecting top-K samples with lowest scores and replicating.

    Args:
        latents: current latent batch (B, dim)
        scores: smoothness scores for each sample (B,)
        top_K: fraction of samples to retain (0 < top_K <= 1)

    Returns:
        arranged_batch: rearranged latent batch with same shape (B, dim)
    """
    B = latents.shape[0]
    num_selected_samples = max(int(top_K * B), 1)

    # Select indices with lowest scores (smoothest inversions)
    topk_indices = torch.topk(scores, k=num_selected_samples, largest=False)[1]

    # Create new batch from selected samples
    arranged_batch = latents[topk_indices].clone()

    # Replicate selected samples to fill batch
    while arranged_batch.shape[0] < B:
        arranged_batch = torch.cat([arranged_batch, arranged_batch], dim=0)

    # Trim to exact batch size
    arranged_batch = arranged_batch[:B]

    return arranged_batch


# =============================================================================
# MAIN CDGS SAMPLER CLASS
# =============================================================================


class CDGS(nn.Module):
    """
    Compositional Diffusion Guidance Sampler (CDGS) for long-horizon planning.

    Automatically detects whether to use:
    - Single model with batched inference (when all model paths are identical)
    - Separate models with per-model inference (when model paths differ)

    Key Features:
    - Compositional prediction: Multiple models cover overlapping windows
    - Iterative resampling: Re-denoise from same timestep for better alignment
    - Inversion pruning: Select samples with smoother inversion paths

    Parameters:
        model_paths: Dictionary of model checkpoint paths for loading models
        device: torch device where models and sampling will run
        model_type: 'diffusion' or 'flow' (flow support is scaffolded for future)
        num_bridges: Number of bridge models defining the compositional task

        num_resampling_steps: Number of times to denoise from each timestep
            - 1 (default): No iterative resampling, standard compositional sampling
            - >1: Enable iterative resampling for better compositional alignment

        enable_pruning: Whether to apply inversion-based pruning
            - False (default): No pruning, standard resampling
            - True: Apply pruning to select smoother samples during generation

        pruning_start: Fraction of sampling steps before pruning starts (0-1)
        pruning_end: Fraction of sampling steps after pruning ends (0-1)
        pruning_top_K: Fraction of batch to retain during pruning (0-1)
    """

    def __init__(
        self,
        model_paths,
        device,
        model_type="diffusion",
        num_bridges=1,
        num_resampling_steps=1,
        enable_pruning=False,
        pruning_start=0.1,
        pruning_end=0.9,
        pruning_top_K=0.2,
    ):
        """Initialize the CDGS sampler with specified configuration."""
        super().__init__()
        self.device = device
        self.model_type = model_type

        # Resampling and pruning parameters
        self.num_resampling_steps = num_resampling_steps
        self.enable_pruning = enable_pruning
        self.pruning_start = pruning_start
        self.pruning_end = pruning_end
        self.pruning_top_K = pruning_top_K

        # Detect if using single or separate models
        unique_paths = set(model_paths.values())
        self.use_single_model = len(unique_paths) == 1

        # Load models
        loaded_models = load_models(device, model_paths, model_type, num_bridges)
        if not loaded_models:
            raise ValueError("Models could not be loaded. Please check model paths.")

        if self.use_single_model:
            # Single unified model for all views
            self.model = loaded_models[0]
            self.num_models = len(loaded_models)
        else:
            # Separate models for each view
            self.models = loaded_models
            self.num_models = len(loaded_models)

        self.views = create_views(self.num_models)
        self.latent_dim = self.views[-1][1]

        # Initialize DDPM scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True,
        )

        # Print configuration
        print("CDGS initialized:")
        print(f"  Model type: {self.model_type}")
        print(
            f"  Model mode: {'single (batched inference)' if self.use_single_model else 'separate (per-model inference)'}"
        )
        print(f"  Number of views: {self.num_models}")
        print(f"  Views: {self.views}")
        print(f"  Latent dimension: {self.latent_dim}")
        print(
            f"  Resampling steps: {self.num_resampling_steps}"
            + (" (disabled)" if self.num_resampling_steps == 1 else "")
        )
        print(f"  Pruning: {'enabled' if self.enable_pruning else 'disabled'}")
        if self.enable_pruning:
            print(f"    - Pruning window: {self.pruning_start} to {self.pruning_end}")
            print(f"    - Top-K fraction: {self.pruning_top_K}")

    def get_compositional_prediction(self, latent, t):
        """
        Compute compositional prediction by averaging overlapping view outputs.

        Args:
            latent: current latent state (B, latent_dim)
            t: current timestep (scalar or tensor)

        Returns:
            combined_pred: compositional prediction (B, latent_dim)
        """
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        B = latent.shape[0]

        # Ensure time is a tensor with shape (B,)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=latent.dtype, device=self.device)
        t_vec = t.reshape(1).repeat(B).to(self.device)

        if self.use_single_model:
            # Batched inference with single model
            batched_views = []
            for start, end in self.views:
                latent_view = latent[:, start:end]
                batched_views.append(latent_view)

            # Stack: (num_views, B, local_dim) -> (num_views * B, local_dim)
            batched_input = torch.cat(batched_views, dim=0)
            batched_t = t_vec.repeat(self.num_models)

            # Single forward pass
            batched_pred = self.model(batched_input, batched_t)

            # Split predictions back
            preds = torch.split(batched_pred, B, dim=0)

            # Distribute to output
            for id, (start, end) in enumerate(self.views):
                value[:, start:end] += preds[id]
                count[:, start:end] += 1
        else:
            # Per-model inference with separate models
            for id, (start, end) in enumerate(self.views):
                latent_view = latent[:, start:end]
                pred = self.models[id](latent_view, t_vec)

                value[:, start:end] += pred
                count[:, start:end] += 1

        # Average predictions in overlapping regions
        combined_pred = torch.where(count > 0, value / count, value)
        return combined_pred

    def inversion_pruning(self, pred_x0, latents, return_scores=False):
        """
        Prune batch by selecting samples with smoothest inversion paths.

        Args:
            pred_x0: predicted clean samples (B, latent_dim)
            latents: current noisy latents (B, latent_dim)
            return_scores: if True, return scores along with rearranged batch

        Returns:
            arranged_batch: rearranged latent batch (B, latent_dim)
            final_scores (optional): inversion scores for each sample (B,)
        """
        B = pred_x0.shape[0]

        # Split predicted clean samples into local views
        batched_x0s = []
        for start, end in self.views:
            batched_x0s.append(pred_x0[:, start:end])
        batched_x0s = torch.stack(batched_x0s, dim=0)  # [num_models, B, local_dim]

        # Compute smoothness scores
        if self.use_single_model:
            final_scores = compute_inversion_scores_single(
                batched_x0s, self.views, self.model, self.scheduler, self.device
            )
        else:
            final_scores = compute_inversion_scores_separate(
                batched_x0s, self.views, self.models, self.scheduler, self.device
            )

        # Rearrange batch by selecting smoothest samples
        arranged_batch = rearrange_batch_by_scores(
            latents, final_scores, self.pruning_top_K
        )

        if return_scores:
            return arranged_batch, final_scores
        return arranged_batch

    def add_noise(self, pred_original_sample, sample, t):
        prev_t = self.scheduler.previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.one
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        if t > 0:
            variance_noise = randn_tensor(
                pred_prev_sample.shape,
                generator=None,
                device=pred_prev_sample.device,
                dtype=pred_prev_sample.dtype,
            )
            variance = (
                self.scheduler._get_variance(t, predicted_variance=None) ** 0.5
            ) * variance_noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    @torch.no_grad()
    def sample(self, batch_size=100, num_inference_steps=100):
        """
        Generate samples using compositional diffusion with optional resampling and pruning.

        Args:
            batch_size: number of trajectories to sample
            num_inference_steps: number of denoising steps

        Returns:
            latent: sampled sequences (batch_size, latent_dim)
        """
        # Initialize from random noise
        sequence_dim = self.views[-1][1]
        latent = torch.randn((batch_size, sequence_dim)).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        num_timesteps = len(self.scheduler.timesteps)

        with torch.autocast(device_type=self.device):
            # Main denoising loop
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                # Iterative resampling loop
                for u in range(self.num_resampling_steps):
                    # DENOISING STEP
                    if self.model_type == SimpleDiffusionModel:
                        # Get compositional noise prediction
                        eps_pred = self.get_compositional_prediction(latent, t)

                        # Take denoising step
                        output = self.scheduler.step(eps_pred, t, latent)
                        pred_x0 = output.pred_original_sample
                        pred_x0[:, 0] = 0
                        pred_x0[:, -1] = 0
                        latent = self.add_noise(pred_x0, latent, t)

                    else:  # flow model
                        raise NotImplementedError(
                            "Flow model sampling not implemented yet. "
                            "Would require ODE/SDE solver instead of DDPM scheduler."
                        )

                    # PRUNING STEP
                    if (
                        self.enable_pruning
                        and u == self.num_resampling_steps - 2
                        and self.num_resampling_steps > 1
                    ):
                        if (
                            self.pruning_start * num_timesteps
                            < i
                            < self.pruning_end * num_timesteps
                        ):
                            latent = self.inversion_pruning(pred_x0, latent)

                    # UNDO STEP
                    if (
                        self.num_resampling_steps > 1
                        and u < self.num_resampling_steps - 1
                    ):
                        if 0 < i < len(self.scheduler.timesteps) - 1:
                            latent = undo_step(latent, t, self.scheduler)

        return latent
