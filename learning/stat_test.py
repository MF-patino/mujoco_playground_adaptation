import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu

# Plotting configurations
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams["legend.handlelength"] = 1.0
plt.rcParams.update({'axes.linewidth': 0.5})

PLOT_DATA_DIR = "plotData"
TRAIN_DATA_SUBDIR = "training"
def main():
    training_path = os.path.join(PLOT_DATA_DIR, TRAIN_DATA_SUBDIR)
    
    # We evaluate the domains where Transfer Learning was applied
    domains = ["RoughTerrain", "SlipperyTerrain", "BlockedKnee"]
    labels = ['Rough Terrain', 'Slippery Terrain', 'Blocked Knee']
    
    samples_per_domain = 20  # You mentioned you ran 20 seeds!
    
    scratch_data_all = []
    adapted_data_all = []
    p_values = []

    print("="*65)
    print("STATISTICAL ANALYSIS REPORT: SCRATCH VS TRANSFER LEARNING")
    print("="*65)

    # ---------------------------------------------------------
    # Dynamic Data Loading & Statistical Testing
    # ---------------------------------------------------------
    for domain in domains:
        scratch_y = []
        adapted_y = []
        
        for i in range(samples_per_domain):
            scratch_file = os.path.join(training_path, f"{domain}_{i}.npz")
            adapted_file = os.path.join(training_path, f"{domain}_{i}_AdaptedFrom_FlatTerrain.npz")
            
            # Load Scratch Data
            if os.path.exists(scratch_file):
                data = np.load(scratch_file)
                scratch_y.append(data['reward_mean'][-1]) # Take final convergence reward
                
            # Load Adapted Data
            if os.path.exists(adapted_file):
                data = np.load(adapted_file)
                adapted_y.append(data['reward_mean'][-1]) # Take final convergence reward
                
        scratch_data_all.append(np.array(scratch_y))
        adapted_data_all.append(np.array(adapted_y))
        
        # Mann-Whitney U test (Two-sided)
        u_stat, p_val = mannwhitneyu(scratch_y, adapted_y, alternative='two-sided')
        p_values.append(p_val)
        
        print(f"Domain: {domain}")
        print(f"  Scratch Mean: {np.mean(scratch_y):.2f} ± {np.std(scratch_y):.2f}")
        print(f"  Adapted Mean: {np.mean(adapted_y):.2f} ± {np.std(adapted_y):.2f}")
        print(f"  Mann-Whitney U: {u_stat:.1f} | p-value: {p_val:.5e}")
        if p_val < 0.05:
            print("  -> Statistically Significant Difference! (p < 0.05)")
        else:
            print("  -> No Significant Difference. (p >= 0.05)")
        print("-" * 65)

    # ---------------------------------------------------------
    # Grouped Boxplot Plotting
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300, facecolor='w', edgecolor='k')
    
    # Calculate X positions for grouped boxplots
    x_indexes = np.arange(len(domains)) * 3
    x_scratch = x_indexes - 0.4
    x_adapted = x_indexes + 0.4

    # Plot Scratch (Red - matches your previous learning curves)
    bp_scratch = ax.boxplot(scratch_data_all, positions=x_scratch, widths=0.6, patch_artist=True)
    for patch, median in zip(bp_scratch['boxes'], bp_scratch['medians']):
        patch.set_facecolor('#F79886') # Light Red
        patch.set_edgecolor('#d62728') # Dark Red
        median.set_color('black')
        median.set_linewidth(2.0)
    for flier in bp_scratch['fliers']:
        flier.set(marker='o', markerfacecolor='#F79886', markeredgecolor='#d62728', markersize=5)

    # Plot Adapted (Blue - matches your previous learning curves)
    bp_adapted = ax.boxplot(adapted_data_all, positions=x_adapted, widths=0.6, patch_artist=True)
    for patch, median in zip(bp_adapted['boxes'], bp_adapted['medians']):
        patch.set_facecolor('#86B3F7') # Light Blue
        patch.set_edgecolor('#1f77b4') # Dark Blue
        median.set_color('black')
        median.set_linewidth(2.0)
    for flier in bp_adapted['fliers']:
        flier.set(marker='o', markerfacecolor='#86B3F7', markeredgecolor='#1f77b4', markersize=5)

    # ---------------------------------------------------------
    # Significance Brackets (P-values)
    # ---------------------------------------------------------
    global_y_max = max(max(np.max(s), np.max(a)) for s, a in zip(scratch_data_all, adapted_data_all))
    global_y_min = min(min(np.min(s), np.min(a)) for s, a in zip(scratch_data_all, adapted_data_all))
    y_range = global_y_max - global_y_min

    for i in range(len(domains)):
        # Find the highest point between the two boxes in this specific domain
        local_max = max(np.max(scratch_data_all[i]), np.max(adapted_data_all[i]))
        
        y_bracket = local_max + (y_range * 0.05)
        h_bracket = y_range * 0.02
        
        # Determine stars
        p_val = p_values[i]
        if p_val < 0.001:
            p_text = "***"
        elif p_val < 0.01:
            p_text = "**"
        elif p_val < 0.05:
            p_text = "*"
        else:
            p_text = "n.s."
            
        # Draw bracket
        x1, x2 = x_scratch[i], x_adapted[i]
        ax.plot([x1, x1, x2, x2], [y_bracket, y_bracket + h_bracket, y_bracket + h_bracket, y_bracket], 
                color='black', linewidth=1.0)
        
        # Add text
        ax.text((x1 + x2) / 2, y_bracket + h_bracket + (y_range * 0.01), p_text, 
                ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(global_y_min - (y_range * 0.05), global_y_max + (y_range * 0.15))
    
    ax.set_title('Final Convergence Reward: Training from Scratch vs. Transfer Learning', pad=15)
    ax.set_ylabel('Mean Episode Reward')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F79886', edgecolor='#d62728', label='Trained from Scratch'),
        Patch(facecolor='#86B3F7', edgecolor='#1f77b4', label='Adapted from Flat Terrain')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.join(os.getcwd(), "01_training_flat_reward"), exist_ok=True)
    save_path = os.path.join(os.getcwd(), "01_training_flat_reward", "01_scratch_vs_adapted_boxplot.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()