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
    
    def _plot_transition_arrows(self, num_indices):
        """
        Plot transition arrows showing valid mode transitions between consecutive indices.
        
        Args:
            num_indices (int): Total number of indices
        """
        transitions = self.config['transitions']
        arrow_settings = self.plotting_settings['transition_arrows']
        
        for i in range(num_indices - 1):
            # Determine transition type based on position
            if i == 0:
                # From start to first intermediate
                transition_type = 'from_start'
            elif i == num_indices - 2:
                # From last intermediate to end
                transition_type = 'to_end'
            else:
                # Between intermediates
                transition_type = 'intermediate'
            
            if transition_type in transitions:
                transition_data = transitions[transition_type]
                
                # Plot arrows for each valid transition
                for transition in transition_data['transitions']:
                    source_y = transition['from']
                    target_y = transition['to']
                    
                    # Calculate arrow positions
                    start_x = i + arrow_settings['offset_x']
                    end_x = (i + 1) - arrow_settings['offset_x']
                    
                    # Draw arrow
                    plt.annotate('', 
                                xy=(end_x, target_y), 
                                xytext=(start_x, source_y),
                                arrowprops=dict(
                                    arrowstyle=f'->, head_length={arrow_settings["head_length"]}, head_width={arrow_settings["head_width"]}', 
                                    color=arrow_settings['color'],
                                    alpha=arrow_settings['alpha'],
                                    lw=arrow_settings['width'] * 100,  # Convert to reasonable linewidth
                                    shrinkA=0, shrinkB=0
                                ))
    
    def plot_overlay(self, csv_file, output_dir="results_long", save_plots=True, 
                     show_data=True, show_connections=True, show_transitions=False):
        """
        Create overlay plot of samples and theoretical distributions.
        
        Args:
            csv_file (str): Path to CSV file with latent data
            output_dir (str): Directory to save plots
            save_plots (bool): Whether to save plots to files
            show_data (bool): Whether to show data scatter points
            show_connections (bool): Whether to show connection lines between points
            show_transitions (bool): Whether to show valid mode transition arrows
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
        
        # Plot scatter points for generated samples if requested
        if show_data:
            for i in range(num_indices):
                plt.scatter(
                    np.ones_like(latents[:, i]) * i, 
                    latents[:, i],
                    label=f'{latent_cols[i]} samples', 
                    alpha=0.5, 
                    s=20
                )
        
        # Plot theoretical Gaussian distributions
        if show_data:
            y_range = (latents.min() - 0.5, latents.max() + 0.5)
        else:
            # Use default range from config when not showing data
            y_range = tuple(self.plotting_settings['x_range'])
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
        
        # Plot connecting lines for a subset of samples if requested
        if show_data and show_connections:
            max_lines = min(50, latents.shape[0] - 1)
            for i in range(max_lines):
                plt.plot(range(num_indices), latents[i], color='gray', alpha=0.1, linewidth=0.5)
        
        # Plot transition arrows if requested
        if show_transitions and 'transitions' in self.config:
            self._plot_transition_arrows(num_indices)
        
        # Format the plot
        if show_data:
            title_parts = [f'Generated samples vs theoretical distributions']
        else:
            title_parts = [f'Theoretical distributions only']
        
        if metadata:
            title_parts.append(f"({metadata.get('mode', 'unknown')} mode, {metadata.get('steps', 'N/A')} steps)")
        
        # plt.title(' '.join(title_parts))
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        plt.ylim(-1.25, 1.25)
        
        # Set LaTeX-formatted x-axis labels with proper math font
        plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern math font
        plt.rcParams['font.family'] = 'serif'    # Use serif font family
        x_labels = [f'$x_{{{i+1}}}$' for i in range(num_indices)]
        plt.xticks(range(num_indices), x_labels, fontsize=60)
        # plt.yticks(fontsize=20)
        
        # Remove plot border/spines and tick lines
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Remove tick lines but keep labels
        ax.tick_params(axis='x', length=0)  # Remove x-axis tick lines
        ax.tick_params(axis='y', length=0, labelleft=False)  # Remove y-axis tick lines and labels
        
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename from CSV file
            csv_name = Path(csv_file).stem
            suffix = "_theory_only" if not show_data else "_overlay"
            if show_transitions:
                suffix += "_transitions"
            output_file = f"{output_dir}/plot_{csv_name}{suffix}.png"
            plt.savefig(output_file, dpi=600, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', 
                       format='png', transparent=False, 
                       pad_inches=0.1, metadata={'Creator': 'Matplotlib'})
            print(f"Plot saved as: {output_file}")
        
        plt.show()
        


def main():
    parser = argparse.ArgumentParser(description='Plot overlay of latent results with theoretical Gaussian distributions')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to CSV file containing latent data')
    parser.add_argument('--config', type=str, default='gaussian_parameters.json',
                        help='Path to JSON configuration file (default: gaussian_parameters.json)')
    parser.add_argument('--output_dir', type=str, default='results_long',
                        help='Output directory for plots (default: results_long)')
    parser.add_argument('--no_save', action='store_true',
                        help='Don\'t save plots to files, only display')
    parser.add_argument('--no_data', action='store_true',
                        help='Show only theoretical distributions without data points')
    parser.add_argument('--no_connections', action='store_true',
                        help='Don\'t show connection lines between data points')
    parser.add_argument('--show_transitions', action='store_true',
                        help='Show valid mode transition arrows between consecutive indices')
    
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
    show_data = not args.no_data
    show_connections = not args.no_connections
    show_transitions = args.show_transitions
    
    # Determine number of indices from CSV
    df = pd.read_csv(args.csv)
    latent_cols = [col for col in df.columns if col.startswith('x')]
    num_indices = len(latent_cols)
    
    print(f"Found {num_indices} latent dimensions in CSV file")
    
    plot_description = []
    if show_data:
        plot_description.append("data")
    plot_description.append("theoretical distributions")
    if show_transitions:
        plot_description.append("transition arrows")
    
    print(f"Generating plot with: {', '.join(plot_description)}...")
    
    plotter.plot_overlay(args.csv, args.output_dir, save_plots, show_data, show_connections, show_transitions)
    
    print("Plotting completed!")


if __name__ == "__main__":
    main()
