from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np

from model import create_simple_diffusion_model

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

class ToyDomainDiffusion(nn.Module):
    def __init__(self, device, num_bridges=4):
        super().__init__()

        self.device = device
        self.model1 = create_simple_diffusion_model()
        self.model2 = create_simple_diffusion_model()
        self.model3 = create_simple_diffusion_model()
        self.models = [self.model1]
        self.models += [self.model2]*num_bridges
        self.models += [self.model3]
        self.models[0].load_state_dict(torch.load("models/simple_diffusion_model_1.pth", map_location=device))
        self.models[-1].load_state_dict(torch.load("models/simple_diffusion_model_3.pth", map_location=device))
        for id, model in enumerate(self.models[1:-1]):
            model.load_state_dict(torch.load(f"models/simple_diffusion_model_2.pth", map_location=device))

        for model in self.models:
            model.to(device)
            model.eval()

        self.views = [(i, i+2) for i in range(len(self.models))]

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )

    
    def undo_step(self, latents, pred_x0, eps_t, timestep, generator=None):
        n = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        for i in range(n):
            beta = self.scheduler.betas[timestep + i]
            if latents.device.type == "mps":
                # randn does not work reproducibly on mps
                noise = randn_tensor(latents.shape, dtype=latents.dtype, generator=generator)
                noise = noise.to(latents.device)
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)

            # 10. Algorithm 1 Line 10 https://huggingface.co/papers/2201.09865
            latents = (1 - beta) ** 0.5 * latents + beta**0.5 * noise

        return latents

    @torch.no_grad()
    def gsc(self, batch_size=100, num_inference_steps=100, device='cuda'):

        total_sample_dim = self.views[-1][1]

        latent = torch.randn((batch_size, total_sample_dim)).to(device) 
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                count.zero_()
                value.zero_()

                for id, (start, end) in enumerate(self.views):
                    # TODO we can support batches, and pass multiple views at once to the unet
                    latent_view = latent[:, start:end]
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    
                    # predict the noise residual
                    noise_pred = self.models[id](latent_view, t.expand(batch_size).to(device))

                    # compute the denoising step with the reference model
                    value[:, start:end] += noise_pred
                    count[:, start:end] += 1

                # take the MultiDiffusion step
                combined_noise = torch.where(count > 0, value / count, value)
                latent = self.scheduler.step(combined_noise, t, latent)['prev_sample']

        return latent
    
    @torch.no_grad()
    def gsc_resample(self, batch_size=100, num_inference_steps=100, num_resampling_steps=10, device='cuda'):

        total_sample_dim = self.views[-1][1]

        latent = torch.randn((batch_size, total_sample_dim)).to(device) 
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                U = int(min(max((float(i) / float(len(self.scheduler.timesteps))) * num_resampling_steps, 5), num_resampling_steps))
                for u in tqdm(range(U), leave=False):
                    count.zero_()
                    value.zero_()
                    for id, (start, end) in enumerate(self.views):
                        # TODO we can support batches, and pass multiple views at once to the unet
                        latent_view = latent[:, start:end]
                        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                        
                        # predict the noise residual
                        noise_pred = self.models[id](latent_view, t.expand(batch_size).to(device))

                        # compute the denoising step with the reference model
                        value[:, start:end] += noise_pred
                        count[:, start:end] += 1

                    # take the MultiDiffusion step
                    combined_noise = torch.where(count > 0, value / count, value)
                    latent = self.scheduler.step(combined_noise, t, latent)

                    if u < U-1 and i < len(self.scheduler.timesteps)-1 and i > 0:
                        pred_x0 = latent['pred_original_sample']
                        latent = latent['prev_sample']
                        latent = self.undo_step(latent, pred_x0, combined_noise, t)
                    else:
                        latent = latent['prev_sample']

        return latent
    
    @torch.no_grad()
    def gsc_resample_pruning(self, batch_size=100, num_inference_steps=100, num_resampling_steps=10, top_K=0.2, device='cuda'):

        total_sample_dim = self.views[-1][1]

        latent = torch.randn((batch_size, total_sample_dim)).to(device) 
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)
        num_timesteps = len(self.scheduler.timesteps)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                U = int(min(max((float(i) / float(len(self.scheduler.timesteps))) * num_resampling_steps, 5), num_resampling_steps))
                for u in tqdm(range(U), leave=False):
                    count.zero_()
                    value.zero_()
                    for id, (start, end) in enumerate(self.views):
                        # TODO we can support batches, and pass multiple views at once to the unet
                        latent_view = latent[:, start:end]
                        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                        
                        # predict the noise residual
                        noise_pred = self.models[id](latent_view, t.expand(batch_size).to(device))

                        # compute the denoising step with the reference model
                        value[:, start:end] += noise_pred
                        count[:, start:end] += 1

                    # take the MultiDiffusion step
                    combined_noise = torch.where(count > 0, value / count, value)
                    latent = self.scheduler.step(combined_noise, t, latent)

                    pred_x0 = latent['pred_original_sample']
                    latent = latent['prev_sample']

                    if u == U-2 and (i < 1.0*num_timesteps and i > 0.0*num_timesteps):
                        latent = self.inversion_pruning(
                            pred_x0,
                            latent,
                            self.views,
                            top_K
                        )

                    if u < U-1 and i < len(self.scheduler.timesteps)-1 and i > 0:
                        latent = self.undo_step(latent, pred_x0, combined_noise, t)

        return latent
    

    def inversion_pruning(self, pred_x0, latents, views, top_K):
        # Implement the inversion pruning logic here
        num_models = len(views)
        B = pred_x0.shape[0]
        all_timesteps = self.scheduler.timesteps.flip(dims=(0,))
        num_timesteps = len(all_timesteps)

        batched_x0s = []
        for start, end in views:
            batched_x0s.append(pred_x0[:, start:end])

        batched_x0s = torch.stack(batched_x0s, dim=0)  # [num_models, B, 2]

        inversion_latents = batched_x0s.clone()
        all_noise_prediction = []

        # for idx, i in tqdm(enumerate(all_timesteps[:-1]), leave=False, total=len(all_timesteps)-1):
        for idx, i in tqdm(enumerate(all_timesteps[:-num_timesteps//2+1]), leave=False, total=num_timesteps//2):
            t = i
            t_next = all_timesteps[idx + 1]
            alpha_t = self.scheduler.alphas_cumprod[t]
            alpha_t_next = self.scheduler.alphas_cumprod[t_next]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)

            with torch.no_grad():
                noise_pred_combined = torch.zeros_like(inversion_latents)
                for id, (start, end) in enumerate(views):
                    latent_view = inversion_latents[id]
                    noise_pred = self.models[id](latent_view, t.expand(B).to(latents.device))
                    noise_pred_combined[id] = noise_pred

                x0_pred = (inversion_latents - sqrt_one_minus_alpha_t * noise_pred_combined) / sqrt_alpha_t
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
                noise_pred_combined = (inversion_latents - sqrt_alpha_t * x0_pred) / sqrt_one_minus_alpha_t
                inversion_latents = sqrt_alpha_t_next * x0_pred + sqrt_one_minus_alpha_t_next * noise_pred_combined
                all_noise_prediction.append(noise_pred_combined)

        all_intermediate_noise_preds = torch.stack(all_noise_prediction, dim=2) # [num_models, B, num_inference_steps, 2]
        derivative = torch.diff(all_intermediate_noise_preds, dim=2)

        all_scores = torch.norm(derivative.reshape(num_models*B, -1), dim=1).reshape(num_models, B)
        final_scores = all_scores.mean(dim=0) # (B,)

        num_selected_samples = max(int(top_K * B), 1)
        topk_indices = torch.topk(final_scores, k = num_selected_samples, largest=False)[1]

        arranged_batch = latents.clone()
        arranged_batch = arranged_batch[topk_indices]

        while arranged_batch.shape[0] < B:
            arranged_batch = torch.cat([arranged_batch, arranged_batch], dim=0)

        arranged_batch = arranged_batch[:B]

        return arranged_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--outfile', type=str, default='results')
    parser.add_argument('--num_resamples', type=int, default=1)
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'resample', 'pruning'],
                        help='inference mode')
    parser.add_argument('--num_bridges', type=int, default=4,
                        help='number of bridge models between the two reference models')

    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = ToyDomainDiffusion(
        device,
        num_bridges=opt.num_bridges
    )

    inference_fn = sd.gsc
    if opt.mode == "resample":
        inference_fn = sd.gsc_resample
    elif opt.mode == "pruning":
        inference_fn = sd.gsc_resample_pruning

    latents = inference_fn(
        batch_size=100,
        num_inference_steps=opt.steps,
        device=device
    )

    latents = latents.cpu().numpy()

    # plot latents[0] at x = 0, latents[1] at x = 1, latents[2] at x = 2, latents[3] at x = 3
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))

    num_xs = len(sd.views) + 1

    for i in range(num_xs):
        plt.scatter(np.ones_like(latents[:, i]) * i, latents[:, i], label=f'x{i+1}', alpha=0.5)

    # plot lines between latents[0] and latents[1], latents[1] and latents[2], latents[2] and latents[3]
    for i in range(latents.shape[0] - 1):
        # plt.plot([0., 1., 2., 3.], [latents[i, 0], latents[i, 1], latents[i, 2], latents[i, 3]], color='gray', alpha=0.2)
        plt.plot(range(num_xs), latents[i], color='gray', alpha=0.2)

    plt.title(f'Generated samples after {opt.steps} steps of {opt.mode} inference')
    plt.xlabel('Dummy X-axis')
    plt.ylabel('Dummy Y-axis')
    plt.legend()
    plt.grid()
    plt.savefig(f"results_long/long_{opt.outfile}_samples_{opt.mode}_{opt.steps}_steps.png")
    plt.close()
