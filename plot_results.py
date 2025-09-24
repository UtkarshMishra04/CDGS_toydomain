#!/usr/bin/env python3
"""
Plotting script to visualize generated latents from CSV files using 
theoretical Gaussian parameters from JSON configuration.

Usage:
    python plot_results.py --csv latents_file.csv --config gaussian_parameters.json
    python plot_results.py --csv latents_file.csv  # Uses default config
    python plot_results.py --help
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
import os
from pathlib import Path


class GaussianPlotter:
    def __init__(self, config_file="gaussian_parameters.json"):
        """
        Initialize the plotter with configuration from JSON file.
        
        Args:
            config_file (str): Path to JSON configuration file
        """
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.params = self.config['parameters']
        self.plotting_settings = self.config['plotting_settings']
        
    def get_gaussian_for_index(self, index, total_indices):
        """
        Get Gaussian parameters for a specific index.
        
        Args:
            index (int): The index position
            total_indices (int): Total number of indices
            
        Returns:
            dict: Gaussian parameters for the index
        """
        if index == 0:
            return self.params['index_0']
        elif index == total_indices - 1:
            return self.params['index_last']
        else:
            return self.params['index_middle']
    
    def compute_pdf(self, y_points, gaussian_params):
        """
        Compute probability density function for given parameters.
        
        Args:
            y_points (np.array): Y values to compute PDF for
            gaussian_params (dict): Gaussian parameters
            
        Returns:
            np.array: PDF values
        """
        if gaussian_params['type'] == 'single_gaussian':
            return norm.pdf(y_points, gaussian_params['mu'], gaussian_params['sigma'])
        
        elif gaussian_params['type'] == 'mixture_gaussian':
            pdf = np.zeros_like(y_points)
            for component in gaussian_params['components']:
                pdf += component['weight'] * norm.pdf(
                    y_points, component['mu'], component['sigma']
                )
            return pdf
        
        else:
            raise ValueError(f"Unknown gaussian type: {gaussian_params['type']}")
    
    def plot_overlay(self, csv_file, output_dir="results_long", save_plots=True):
        """
        Create overlay plot of samples and theoretical distributions.
        
        Args:
            csv_file (str): Path to CSV file with latent data
            output_dir (str): Directory to save plots
            save_plots (bool): Whether to save plots to files
        """
        # Load data
        df = pd.read_csv(csv_file)
        
        # Extract latent columns (x1, x2, x3, etc.)
        latent_cols = [col for col in df.columns if col.startswith('x')]
        latents = df[latent_cols].values
        num_indices = len(latent_cols)
        
        # Extract metadata for plot title
        metadata = {}
        for col in ['mode', 'steps', 'seed', 'num_bridges']:
            if col in df.columns:
                metadata[col] = df[col].iloc[0]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot scatter points for generated samples
        for i in range(num_indices):
            plt.scatter(
                np.ones_like(latents[:, i]) * i, 
                latents[:, i],
                label=f'{latent_cols[i]} samples', 
                alpha=0.5, 
                s=20
            )
        
        # Plot theoretical Gaussian distributions
        y_range = (latents.min() - 0.5, latents.max() + 0.5)
        y_points = np.linspace(y_range[0], y_range[1], self.plotting_settings['num_points'])
        
        for i in range(num_indices):
            gaussian_params = self.get_gaussian_for_index(i, num_indices)
            pdf = self.compute_pdf(y_points, gaussian_params)
            
            # Scale and position the PDF
            pdf_scaled = i + pdf * self.plotting_settings['pdf_scale_factor']
            baseline = np.full_like(pdf_scaled, i)  # Create baseline at index position
            
            # Choose color based on gaussian type
            if gaussian_params['type'] == 'single_gaussian':
                if i == 0:
                    color = self.plotting_settings['colors']['index_0']
                    label = f'{latent_cols[i]} theory (μ={gaussian_params["mu"]}, σ={gaussian_params["sigma"]})'
                else:
                    color = self.plotting_settings['colors']['index_last']
                    label = f'{latent_cols[i]} theory (μ={gaussian_params["mu"]}, σ={gaussian_params["sigma"]})'
            else:
                color = self.plotting_settings['colors']['index_middle']
                label = f'{latent_cols[i]} theory (mixture)'
            
            # Create filled area instead of line plot
            plt.fill_betweenx(
                y_points, baseline, pdf_scaled,
                color=color,
                alpha=0.3,  # Semi-transparent fill
                label=label,
                edgecolor=color,
                linewidth=1
            )
        
        # Plot connecting lines for a subset of samples
        max_lines = min(50, latents.shape[0] - 1)
        for i in range(max_lines):
            plt.plot(range(num_indices), latents[i], color='gray', alpha=0.1, linewidth=0.5)
        
        # Format the plot
        title_parts = [f'Generated samples vs theoretical distributions']
        if metadata:
            title_parts.append(f"({metadata.get('mode', 'unknown')} mode, {metadata.get('steps', 'N/A')} steps)")
        
        plt.title(' '.join(title_parts))
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename from CSV file
            csv_name = Path(csv_file).stem
            output_file = f"{output_dir}/plot_{csv_name}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Overlay plot saved as: {output_file}")
        
        plt.show()
        
    def plot_separate_gaussians(self, num_indices, output_dir="results_long", save_plots=True):
        """
        Create separate plots for each Gaussian distribution.
        
        Args:
            num_indices (int): Number of indices to plot
            output_dir (str): Directory to save plots
            save_plots (bool): Whether to save plots to files
        """
        x_range = self.plotting_settings['x_range']
        x_points = np.linspace(x_range[0], x_range[1], self.plotting_settings['num_points'])
        
        # Create subplot grid
        cols = min(4, num_indices)
        rows = (num_indices + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        
        # Handle different subplot configurations
        if num_indices == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten() if num_indices > 1 else axes
        
        for i in range(num_indices):
            ax = axes_flat[i]
            gaussian_params = self.get_gaussian_for_index(i, num_indices)
            pdf = self.compute_pdf(x_points, gaussian_params)
            
            if gaussian_params['type'] == 'single_gaussian':
                ax.plot(x_points, pdf, linewidth=2, 
                       label=f"μ={gaussian_params['mu']}, σ={gaussian_params['sigma']}")
                ax.fill_between(x_points, pdf, alpha=0.3)
                title = f'Index {i}: Single Gaussian'
                
            else:  # mixture
                # Plot individual components
                for j, component in enumerate(gaussian_params['components']):
                    comp_pdf = component['weight'] * norm.pdf(
                        x_points, component['mu'], component['sigma']
                    )
                    ax.plot(x_points, comp_pdf, '--', alpha=0.7,
                           label=f"μ={component['mu']}, σ={component['sigma']}")
                
                # Plot mixture
                ax.plot(x_points, pdf, linewidth=2, label='Mixture')
                ax.fill_between(x_points, pdf, alpha=0.3)
                title = f'Index {i}: Gaussian Mixture'
            
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Value')
            ax.set_ylabel('Probability Density')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
        
        # Hide empty subplots
        for i in range(num_indices, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/separate_gaussians_{num_indices}_indices.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Separate Gaussians plot saved as: {output_file}")
        
        plt.show()
    
    def plot_combined_gaussians(self, num_indices, output_dir="results_long", save_plots=True):
        """
        Create combined plot of all Gaussian distributions.
        
        Args:
            num_indices (int): Number of indices to plot
            output_dir (str): Directory to save plots
            save_plots (bool): Whether to save plots to files
        """
        x_range = self.plotting_settings['x_range']
        x_points = np.linspace(x_range[0], x_range[1], self.plotting_settings['num_points'])
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, num_indices))
        
        plt.figure(figsize=(12, 8))
        
        for i in range(num_indices):
            gaussian_params = self.get_gaussian_for_index(i, num_indices)
            pdf = self.compute_pdf(x_points, gaussian_params)
            
            if gaussian_params['type'] == 'single_gaussian':
                label = f'Index {i}: μ={gaussian_params["mu"]}, σ={gaussian_params["sigma"]}'
            else:
                label = f'Index {i}: Mixture'
            
            plt.plot(x_points, pdf, color=colors[i], linewidth=2, label=label)
        
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(f'Comparison of {num_indices} 1D Gaussian Distributions', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/combined_gaussians_{num_indices}_indices.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Combined Gaussians plot saved as: {output_file}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot latent results with theoretical Gaussian distributions')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to CSV file containing latent data')
    parser.add_argument('--config', type=str, default='gaussian_parameters.json',
                        help='Path to JSON configuration file (default: gaussian_parameters.json)')
    parser.add_argument('--output_dir', type=str, default='results_long',
                        help='Output directory for plots (default: results_long)')
    parser.add_argument('--no_save', action='store_true',
                        help='Don\'t save plots to files, only display')
    parser.add_argument('--plot_type', type=str, default='all', 
                        choices=['overlay', 'separate', 'combined', 'all'],
                        help='Type of plots to generate (default: all)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found")
        return
    
    # Initialize plotter
    plotter = GaussianPlotter(args.config)
    save_plots = not args.no_save
    
    # Determine number of indices from CSV
    df = pd.read_csv(args.csv)
    latent_cols = [col for col in df.columns if col.startswith('x')]
    num_indices = len(latent_cols)
    
    print(f"Found {num_indices} latent dimensions in CSV file")
    print(f"Generating plots...")
    
    # Generate requested plots
    if args.plot_type in ['overlay', 'all']:
        print("\nGenerating overlay plot...")
        plotter.plot_overlay(args.csv, args.output_dir, save_plots)
    
    if args.plot_type in ['separate', 'all']:
        print("\nGenerating separate Gaussians plot...")
        plotter.plot_separate_gaussians(num_indices, args.output_dir, save_plots)
    
    if args.plot_type in ['combined', 'all']:
        print("\nGenerating combined Gaussians plot...")
        plotter.plot_combined_gaussians(num_indices, args.output_dir, save_plots)
    
    print("\nPlotting completed!")


if __name__ == "__main__":
    main()
