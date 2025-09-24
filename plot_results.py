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
    
    def _is_within_2std(self, value, gaussian_params, num_stds=3):
        """
        Check if a value is within 2 standard deviations of the theoretical distribution.
        
        Args:
            value (float): The value to check
            gaussian_params (dict): Gaussian parameters for the distribution
            
        Returns:
            bool: True if within 2 standard deviations, False otherwise
        """
        if gaussian_params['type'] == 'single_gaussian':
            mu = gaussian_params['mu']
            sigma = gaussian_params['sigma']
            return abs(value - mu) <= num_stds * sigma
        
        elif gaussian_params['type'] == 'mixture_gaussian':
            # For mixture, check if within 2 std of ANY component
            # This is the correct approach for mixture distributions
            for component in gaussian_params['components']:
                mu = component['mu']
                sigma = component['sigma']
                if abs(value - mu) <= num_stds * sigma:
                    return True
            return False
        
        else:
            return False
    
    def _get_closest_mode(self, value, gaussian_params):
        """
        Get the closest mode center for a given value.
        
        Args:
            value (float): The value to find closest mode for
            gaussian_params (dict): Gaussian parameters
            
        Returns:
            float: The center of the closest mode
        """
        if gaussian_params['type'] == 'single_gaussian':
            return gaussian_params['mu']
        
        elif gaussian_params['type'] == 'mixture_gaussian':
            # Find the closest component center
            closest_mu = None
            min_distance = float('inf')
            
            for component in gaussian_params['components']:
                distance = abs(value - component['mu'])
                if distance < min_distance:
                    min_distance = distance
                    closest_mu = component['mu']
            
            return closest_mu
        
        return None
    
    def _is_valid_transition(self, start_value, end_value, start_idx, end_idx, num_indices):
        """
        Check if a transition between two values follows the allowed transition rules.
        
        Args:
            start_value (float): Starting value
            end_value (float): Ending value  
            start_idx (int): Starting index
            end_idx (int): Ending index
            num_indices (int): Total number of indices
            
        Returns:
            bool: True if transition is valid according to rules
        """
        if 'transitions' not in self.config:
            return True  # No transition rules defined, allow all
        
        # Get the closest modes for start and end values
        start_params = self.get_gaussian_for_index(start_idx, num_indices)
        end_params = self.get_gaussian_for_index(end_idx, num_indices)
        
        start_mode = self._get_closest_mode(start_value, start_params)
        end_mode = self._get_closest_mode(end_value, end_params)
        
        # Determine transition type
        if start_idx == 0:
            transition_type = 'from_start'
        elif end_idx == num_indices - 1:
            transition_type = 'to_end'
        else:
            transition_type = 'intermediate'
        
        # Check if this transition is allowed
        if transition_type in self.config['transitions']:
            allowed_transitions = self.config['transitions'][transition_type]['transitions']
            
            for allowed in allowed_transitions:
                if (abs(start_mode - allowed['from']) < 1e-6 and 
                    abs(end_mode - allowed['to']) < 1e-6):
                    return True
        
        return False
    
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
            green_segments = 0
            red_segments = 0
            
            for i in range(max_lines):
                # Plot each segment individually with appropriate color
                for j in range(num_indices - 1):
                    start_value = latents[i, j]
                    end_value = latents[i, j + 1]
                    
                    # Check if both start and end points are within their expected distributions
                    start_params = self.get_gaussian_for_index(j, num_indices)
                    end_params = self.get_gaussian_for_index(j + 1, num_indices)
                    
                    start_valid = self._is_within_2std(start_value, start_params)
                    end_valid = self._is_within_2std(end_value, end_params)
                    
                    # Check if the transition follows the allowed transition rules
                    transition_valid = self._is_valid_transition(start_value, end_value, j, j + 1, num_indices)
                    
                    # Color segment green only if both endpoints are valid AND transition is allowed
                    segment_color = 'green' if (start_valid and end_valid and transition_valid) else 'red'
                    
                    if start_valid and end_valid and transition_valid:
                        green_segments += 1
                    else:
                        red_segments += 1
                    
                    # Plot individual segment
                    plt.plot([j, j + 1], [start_value, end_value], 
                            color=segment_color, alpha=0.3, linewidth=1)
            
            total_segments = green_segments + red_segments
            print(f"Connection segments: {green_segments} green, {red_segments} red (out of {total_segments} total)")
        
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
