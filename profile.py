import argparse
import torch
import numpy as np
from data import MultiModalDataset
from models import SimpleDiffusionModel
from samplers import CDGS
import time
import os
import json

# Configure device between CUDA, MPS, and CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

num_samples = 1000
seed = 42

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
    num_samples=num_samples,
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
# Test diffusion model loading
diffusion_model_paths = {
    "start": "./pretrained_checkpoints/simple_diffusion_model_start.pth",
    "bridge": "./pretrained_checkpoints/simple_diffusion_model_bridge.pth",
    "end": "./pretrained_checkpoints/simple_diffusion_model_end.pth",
}


def main(args):
    sampler = CDGS(
        model_paths=diffusion_model_paths,
        device=str(device),
        model_type=SimpleDiffusionModel,
        num_bridges=args.num_bridges,
        num_resampling_steps=args.num_resampling_steps,
        pruning_start=args.pruning_start,
        pruning_end=args.pruning_end,
        pruning_top_K=args.pruning_top_K,
        enable_pruning=True,
    )

    start = time.monotonic()
    samples = sampler.sample(
        batch_size=args.num_samples_to_generate,
        num_inference_steps=args.num_inference_steps,
    )
    time_taken = time.monotonic() - start
    print(f"Time taken: {time_taken} seconds")

    fig = MultiModalDataset.plot_multi_step_transitions(
        [start_dataset] + [bridge_dataset] * args.num_bridges + [end_dataset],
        samples.cpu().numpy(),
        annotate_valid=True,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_directory, exist_ok=True)
    fig.savefig(
        os.path.join(
            args.output_directory,
            f"{timestamp}.png",
        )
    )
    with open(
        os.path.join(
            args.output_directory,
            f"{timestamp}.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "time": time_taken,
                "samples_shape": samples.shape,
                "enable_pruning": args.enable_pruning,
                "pruning_start": args.pruning_start,
                "pruning_end": args.pruning_end,
                "pruning_top_K": args.pruning_top_K,
                "num_resampling_steps": args.num_resampling_steps,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples-to-generate", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--num-bridges", type=int, default=5)

    parser.add_argument("--pruning-start", type=float, default=0.0)
    parser.add_argument("--enable-pruning", type=bool, default=False)
    parser.add_argument("--pruning-end", type=float, default=1.0)
    parser.add_argument("--pruning-top-K", type=float, default=0.2)
    parser.add_argument("--num-resampling-steps", type=int, default=10)

    parser.add_argument("--output_directory", type=str, default="./profile")
    args = parser.parse_args()
    main(args)
