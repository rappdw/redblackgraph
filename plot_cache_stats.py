#!/usr/bin/env python3
"""Plot graph statistics from cache metadata files."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot graph statistics from cache files")
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default="/mnt/nas/data/rbg/rappdw-01-13-22",
        help="Directory containing cache JSON files (default: %(default)s)"
    )
    args = parser.parse_args()
    
    cache_dir = Path(args.directory)
    
    # Find all cache JSON files
    json_files = sorted(cache_dir.glob("rappdw_hops*.json"))
    
    if not json_files:
        print(f"No cache JSON files found in {cache_dir}")
        return
    
    # Extract data
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            metadata = json.load(f)
            if 'hops' in metadata and 'vertices' in metadata and 'edges' in metadata:
                data.append({
                    'hops': metadata['hops'],
                    'vertices': metadata['vertices'],
                    'edges': metadata['edges']
                })
    
    if not data:
        print("No valid cache metadata found")
        return
    
    # Sort by hop count
    data.sort(key=lambda x: x['hops'])
    
    # Extract arrays for plotting
    hops = [d['hops'] for d in data]
    vertices = [d['vertices'] for d in data]
    edges = [d['edges'] for d in data]
    
    # Print summary table
    print(f"\n{'Hops':<6} {'Vertices':>12} {'Edges':>12} {'Edges/Vertex':>12}")
    print("-" * 46)
    for d in data:
        ratio = d['edges'] / d['vertices'] if d['vertices'] > 0 else 0
        print(f"{d['hops']:<6} {d['vertices']:>12,} {d['edges']:>12,} {ratio:>12.2f}")
    
    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: Edges vs Vertices Relationship")
    print("=" * 70)
    
    vertices_arr = np.array(vertices)
    edges_arr = np.array(edges)
    
    # Pearson correlation
    corr, p_value = stats.pearsonr(vertices_arr, edges_arr)
    print(f"\nPearson correlation coefficient: {corr:.8f}")
    print(f"P-value: {p_value:.2e}")
    
    # Linear regression: edges = a * vertices + b
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(vertices_arr, edges_arr)
    print(f"\n--- Linear Model: edges = a * vertices + b ---")
    print(f"Slope (a):          {slope:.8f}")
    print(f"Intercept (b):      {intercept:.2f}")
    print(f"R² (coefficient of determination): {r_value**2:.8f}")
    print(f"Standard error:     {std_err:.2e}")
    
    # Compute residuals
    predicted_edges = slope * vertices_arr + intercept
    residuals = edges_arr - predicted_edges
    rmse = np.sqrt(np.mean(residuals**2))
    max_error = np.max(np.abs(residuals))
    mean_abs_error = np.mean(np.abs(residuals))
    
    print(f"\nModel fit quality:")
    print(f"RMSE:               {rmse:,.2f} edges")
    print(f"Mean absolute error: {mean_abs_error:,.2f} edges")
    print(f"Max absolute error:  {max_error:,.2f} edges")
    print(f"Mean relative error: {mean_abs_error / np.mean(edges_arr) * 100:.4f}%")
    
    # Simplified equation (for larger graphs, intercept is negligible)
    print(f"\n--- Simplified Equation (for large graphs) ---")
    print(f"edges ≈ {slope:.4f} * vertices")
    print(f"      ≈ vertices + {(slope-1)*100:.2f}% * vertices")
    
    # Try power law fit: edges = a * vertices^b
    log_v = np.log(vertices_arr)
    log_e = np.log(edges_arr)
    slope_log, intercept_log, r_value_log, _, _ = stats.linregress(log_v, log_e)
    a_power = np.exp(intercept_log)
    b_power = slope_log
    
    print(f"\n--- Power Law Model: edges = a * vertices^b ---")
    print(f"a = {a_power:.6f}")
    print(f"b = {b_power:.6f}")
    print(f"R² = {r_value_log**2:.8f}")
    
    if r_value**2 > r_value_log**2:
        print(f"\n✓ Linear model provides better fit (R² = {r_value**2:.8f})")
    else:
        print(f"\n✓ Power law model provides better fit (R² = {r_value_log**2:.8f})")
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    # Plot 1: Vertices and Edges vs Hop Count (linear scale)
    ax1.plot(hops, vertices, '-', label='Vertices', linewidth=1.5)
    ax1.plot(hops, edges, '-', label='Edges', linewidth=1.5)
    ax1.set_xlabel('Hop Count', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Graph Growth by Hop Count', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(hops)
    
    # Format y-axis with commas
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Plot 2: Log scale
    ax2.semilogy(hops, vertices, '-', label='Vertices', linewidth=1.5)
    ax2.semilogy(hops, edges, '-', label='Edges', linewidth=1.5)
    ax2.set_xlabel('Hop Count', fontsize=12)
    ax2.set_ylabel('Count (log scale)', fontsize=12)
    ax2.set_title('Graph Growth by Hop Count (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xticks(hops)
    
    # Plot 3: Edges vs Vertices with regression line
    ax3.scatter(vertices_arr, edges_arr, alpha=0.6, s=50, label='Actual data')
    v_range = np.linspace(vertices_arr.min(), vertices_arr.max(), 100)
    ax3.plot(v_range, slope * v_range + intercept, 'r-', linewidth=2, 
             label=f'Linear fit: E = {slope:.4f}V + {intercept:.0f}')
    ax3.set_xlabel('Vertices', fontsize=12)
    ax3.set_ylabel('Edges', fontsize=12)
    ax3.set_title(f'Edges vs Vertices (R² = {r_value**2:.6f})', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Format axes with commas
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    # Save the figure
    output_file = cache_dir / "graph_growth.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
