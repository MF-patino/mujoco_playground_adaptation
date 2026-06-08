
import matplotlib, os
try:
    matplotlib.use('TkAgg') 
except:
    print("Warning: Running in headless environment.")
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import numpy as np
import jax.numpy as jp
from matplotlib.patches import Patch
from sklearn.decomposition import TruncatedSVD
from matplotlib.lines import Line2D
import umap

PLOT_DATA_DIR = "plotData"
TRAIN_DATA_SUBDIR = "training"

def plotTransferLearningAggregated(env_names, limit_evals=75, num_trials=20):
    plot_dir = os.path.join(PLOT_DATA_DIR, TRAIN_DATA_SUBDIR)
    
    # Create subplots: 1 row, N columns
    fig, axes = plt.subplots(1, len(env_names), figsize=(5.5 * len(env_names), 5), layout="constrained")
    
    if len(env_names) == 1:
        axes = [axes]
        
    for ax, env_name in zip(axes, env_names):
        scratch_matrix = np.zeros((0, limit_evals))
        transfer_matrix = np.zeros((0, limit_evals))
        
        common_steps = None
        
        for i in range(num_trials):
            scratch_file = os.path.join(plot_dir, f"{env_name}_{i}.npz")
            if os.path.exists(scratch_file):
                data = np.load(scratch_file)
                steps = data['steps']
                means = data['reward_mean']
                
                # Capture the X-axis steps (assuming all runs evaluate at the same intervals)
                if common_steps is None and len(steps) >= limit_evals:
                    common_steps = steps[:limit_evals]
                
                # Forward-fill if stopped early
                length = min(len(means), limit_evals)
                scratch_matrix = np.vstack([scratch_matrix, [means[:length]]])
            else:
                print(f"Warning: {scratch_file} not found.")
                
            transfer_file = os.path.join(plot_dir, f"{env_name}_{i}_AdaptedFrom_FlatTerrain.npz")
            if os.path.exists(transfer_file):
                data = np.load(transfer_file)
                steps = data['steps']
                means = data['reward_mean']
                
                if common_steps is None and len(steps) >= limit_evals:
                    common_steps = steps[:limit_evals]
                    
                # Forward-fill if stopped early
                length = min(len(means), limit_evals)
                transfer_matrix = np.vstack([transfer_matrix, [means[:length]]])
            else:
                print(f"Warning: {transfer_file} not found.")
        
        # Fallback if no run reached limit_evals (assume 1M step intervals)
        if common_steps is None:
            common_steps = np.arange(limit_evals) * 1_000_000 
            
        # --- 3. Compute Aggregated Statistics ---
        scratch_mean = np.mean(scratch_matrix, axis=0)
        scratch_std = np.std(scratch_matrix, axis=0)
        
        transfer_mean = np.mean(transfer_matrix, axis=0)
        transfer_std = np.std(transfer_matrix, axis=0)
        
        # --- 4. Plotting ---
        # Transfer Learning (Blue)
        ax.plot(common_steps, transfer_mean, label="Adapted from Flat", color='#1f77b4', linewidth=2.5)
        ax.fill_between(common_steps, transfer_mean - transfer_std, transfer_mean + transfer_std, 
                        alpha=0.25, color='#1f77b4', edgecolor='none')
        
        # From Scratch (Red)
        ax.plot(common_steps, scratch_mean, label="Trained from Scratch", color='#d62728', linewidth=2.5)
        ax.fill_between(common_steps, scratch_mean - scratch_std, scratch_mean + scratch_std, 
                        alpha=0.25, color='#d62728', edgecolor='none')
        
        # Formatting
        ax.set_title(f"{env_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Environment Steps", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Scientific notation for X-axis (e.g., 2x10^7)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    # Global Formatting
    axes[0].set_ylabel("Mean Episode Reward", fontsize=12)
    
    # Place legend inside the first or last plot, or outside
    axes[-1].legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    
    plt.show()

def policyEmbeddings2D(controller):

    print("Generating 2D Latent Space Plot...")

    # 1. Reduce the raw inaffinity matrix strictly to 2D
    reducer_2d = TruncatedSVD(n_components=2)
    raw_mat_np = np.array(controller.inaffinity_matrix)
    coords_2d = reducer_2d.fit_transform(raw_mat_np)
    variance_ratio = sum(reducer_2d.explained_variance_ratio_) * 100
    print(f"SVD preserved {variance_ratio:.2f}% of the variance.")

    # 2. L2 Normalize the 2D coordinates so they lie on a unit circle
    # This makes the plot perfectly reflect the Cosine Distances used by the GP
    norms = np.linalg.norm(coords_2d, axis=1, keepdims=True)
    coords_2d_norm = coords_2d / norms

    # 3. Plotting Setup
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get a distinct colormap for the policies
    colors = cm.get_cmap('tab10', len(controller.pol_names))

    # 4. Scatter and Annotate each policy
    for i, env_name in enumerate(controller.pol_names):
        x, y = coords_2d_norm[i, 0], coords_2d_norm[i, 1]
        
        # Draw the point
        ax.scatter(x, y, color=colors(i), s=150, edgecolor='black', zorder=3, label=env_name)
        
        # Add the text label slightly offset from the point
        ax.annotate(env_name, (x, y), xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 5. Draw the Unit Circle (represents Cosine Distance = 1.0 boundary)
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', alpha=0.5, zorder=1)
    ax.add_patch(circle)

    # 6. Formatting
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, zorder=1)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, zorder=1)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    
    # Force the aspect ratio to be perfectly square so the circle isn't warped
    ax.set_aspect('equal', adjustable='box')
    
    # Add some padding around the circle so labels fit
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)

    plt.title("Policy Embeddings (2D SVD Projection)", fontsize=14, pad=15)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    
    # Move legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Policies")
    plt.tight_layout()
    
    plt.show()

def policyEmbeddings3D(controller):
    import re # Para extraer los índices numéricos de las políticas

    print("Generating 3D Latent Space Sphere Plot...")

    # Reduce the raw inaffinity matrix strictly to 3D
    # (Make sure we have at least 3 policies to do a 3D projection)
    n_components = min(3, controller.inaffinity_matrix.shape[1])
    reducer_3d = TruncatedSVD(n_components=n_components)
    raw_mat_np = np.array(controller.inaffinity_matrix)
    coords_3d = reducer_3d.fit_transform(raw_mat_np)
    variance_ratio = sum(reducer_3d.explained_variance_ratio_) * 100
    print(f"SVD preserved {variance_ratio:.2f}% of the variance.")

    # Pad with zeros if we somehow have fewer than 3 dimensions
    if coords_3d.shape[1] < 3:
        coords_3d = np.pad(coords_3d, ((0, 0), (0, 3 - coords_3d.shape[1])), mode='constant')

    # L2 Normalize the 3D coordinates so they lie exactly on a 3D Unit Sphere
    norms = np.linalg.norm(coords_3d, axis=1, keepdims=True)
    coords_3d_norm = coords_3d / (norms + 1e-8)

    # Figure Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- ASIGNACIÓN DE COLORES POR DOMINIO ---
    base_names = [name.split("_AdaptedFrom_")[0] for name in controller.pol_names]
    unique_base_names = list(set(base_names))
    cmap = cm.get_cmap('tab10', len(unique_base_names))
    domain_colors = {domain: cmap(i) for i, domain in enumerate(unique_base_names)}

    # --- ASIGNACIÓN DE FORMAS DENTRO DE CADA DOMINIO ---
    # Agrupamos las políticas que comparten el mismo dominio base
    policies_by_domain = {}
    for pol_name in controller.pol_names:
        base_name = pol_name.split("_AdaptedFrom_")[0]
        if base_name not in policies_by_domain:
            policies_by_domain[base_name] = []
        policies_by_domain[base_name].append(pol_name)
    
    # Secuencia de marcadores para las redundancias (empezando con círculo 'o')
    shape_cycle = ['o', '^', 's', 'D', '*', 'p', 'P', 'X', 'h', 'v', '<', '>']
    policy_shapes = {}
    for base_name, pol_list in policies_by_domain.items():
        # Las ordenamos alfabéticamente para que las formas sean consistentes en cada ejecución
        pol_list_sorted = sorted(pol_list)
        for idx, pol_name in enumerate(pol_list_sorted):
            shape_idx = idx % len(shape_cycle)
            policy_shapes[pol_name] = shape_cycle[shape_idx]

    # --- FORMATEADOR LATEX PARA LA TESIS ---
    def format_thesis_label(raw_name):
        def clean_part(part):
            part_lower = part.lower()
            if "flat" in part_lower:
                return "flat"
            elif "rough" in part_lower:
                return "rough"
            elif "slippery" in part_lower:
                return "slippery"
            elif "blocked" in part_lower or "broken" in part_lower or "knee" in part_lower:
                return "blocked"
            return part

        # Extrae de forma segura el índice final (ej: "_1", "_1_4") si existe
        suffix_match = re.search(r'_(\d+(?:_\d+)?)$', raw_name)
        suffix = f"^{{({suffix_match.group(1).replace('_', '-')})}}" if suffix_match else ""

        if "_AdaptedFrom_" in raw_name:
            parts = raw_name.split("_AdaptedFrom_")
            target = clean_part(parts[0])
            original = clean_part(parts[1])
            return f"$\\pi_{{{target}, {original}}}{suffix}$"
        else:
            target = clean_part(raw_name)
            return f"$\\pi_{{{target}}}{suffix}$"

    # Draw the translucent 3D Unit Sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='whitesmoke', alpha=0.15, edgecolor='none')
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)

    # Scatter and Annotate each policy
    for i, env_name in enumerate(controller.pol_names):
        x, y, z = coords_3d_norm[i, 0], coords_3d_norm[i, 1], coords_3d_norm[i, 2]
        base_name = env_name.split("_AdaptedFrom_")[0]
        c = domain_colors[base_name]
        
        # Obtenemos el marcador correspondiente para esta redundancia
        shape = policy_shapes[env_name]
        
        # Draw the point on the sphere surface using its unique shape
        ax.scatter(x, y, z, color=c, marker=shape, s=150, edgecolor='black', depthshade=True)
        
        # Draw a faint line from the origin to the point (shows the vector)
        ax.plot([0, x], [0, y], [0, z], color=c, linestyle='--', alpha=0.6, linewidth=1.5)


    # Formatting
    # Draw origin axes
    ax.plot([-1.2, 1.2],[0, 0], [0, 0], color='gray', linestyle='-', linewidth=0.5)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='gray', linestyle='-', linewidth=0.5)
    ax.plot([0, 0],[0, 0], [-1.2, 1.2], color='gray', linestyle='-', linewidth=0.5)

    # Force perfectly cubic proportions so the sphere isn't squashed into an ellipsoid
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

    ax.set_xlabel("Principal Component 1", labelpad=10)
    ax.set_ylabel("Principal Component 2", labelpad=10)
    ax.set_zlabel("Principal Component 3", labelpad=10)
    plt.title("Latent Policy Space (3D SVD Projection on Unit Sphere)", fontsize=14, pad=20)
    
    # --- CONSTRUCCIÓN DE LEYENDA INDIVIDUAL CON MARCADORES ---
    # Creamos un elemento de leyenda para cada política combinando su forma, color y etiqueta matemática
    legend_elements = [
        Line2D([0], [0], marker=policy_shapes[name], color='w', markerfacecolor=domain_colors[name.split("_AdaptedFrom_")[0]], 
               markersize=10, markeredgecolor='black', label=format_thesis_label(name))
        for name in sorted(controller.pol_names)
    ]
    
    # Renderizamos la leyenda en el lateral derecho de la figura con el tamaño de letra correcto
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
              title="Policies", title_fontsize=13, fontsize=12, framealpha=0.9)
    plt.tight_layout()
    
    plt.show()

def wmErrorHistory(controller, env_change = None):
    if env_change is None:
        env_change = controller.env_changes[-1]

    extra_steps = 250
    last_env_change, env_name = env_change
    start_step = max(last_env_change - extra_steps, 0)
    end_step = last_env_change + extra_steps

    i = controller.env_changes.index(env_change)
    prev_env = 'None' if i == 0 else controller.env_changes[i-1][1]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axvline(x=last_env_change - start_step, color='green', linestyle='--', linewidth=2, zorder=5)

    legend_elements = []
    for wm_name in controller.smooth_errors:
        legend_elements.append(ax.plot(controller.smooth_errors[wm_name][start_step:end_step], label=f"{wm_name} WM errors")[0])

    legend_elements.append(Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Change: {prev_env} -> {env_name}'))
    
    # Place legend outside the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), title="Legend")
    plt.xlabel("Time step")
    plt.title("WM error history")
    plt.tight_layout()
    plt.show()

def plotGaitPattern(controller, env_change = None):
    if env_change is None:
        env_change = controller.env_changes[-1]

    extra_steps = 250
    last_env_change, env_name = env_change
    start_step = max(last_env_change - 125, 0)
    end_step = last_env_change + extra_steps + 125

    print("Generating Gait Pattern Plot...")

    # Convert list of boolean arrays to a 2D numpy array (Time x 4)
    contacts = np.array(controller.contact_history)
    
    # The Y-axis will go from 0 (bottom) to 3 (top)
    feet_names = ["RR", "RL", "FR", "FL"] 
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # 1. Plot the foot contacts (Converted to Absolute Seconds)
    for i in range(len(feet_names)):
        # Invert the index so FR (idx 0) is at the top (y=3)
        y_level = 3 - i 
        contact_bools = contacts[start_step:end_step, i]
        
        # Find the start and end indices of continuous contact segments
        padded = np.pad(contact_bools, (1, 1), mode='constant', constant_values=False)
        diffs = np.diff(padded.astype(int))
        
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        # Convert indices to absolute time in seconds (t_abs = (idx + start_step) / 50 Hz)
        xranges = [
            ((start + start_step) / 50.0, (end - start) / 50.0) 
            for start, end in zip(starts, ends)
        ]
        
        # Plot solid rectangles
        ax.broken_barh(xranges, (y_level - 0.2, 0.4), facecolors='black', zorder=4)

    # --- UNIQUE COLOR FOR EACH DISTINCT POLICY ---
    unique_policies = list(controller.pol_names)
    cmap = cm.get_cmap('tab10_r' if len(unique_policies) <= 10 else 'tab20', len(unique_policies))
    policy_colors = {pol: cmap(i) for i, pol in enumerate(unique_policies)}
    
    start_idx = 0
    recent_policy_history = controller.policy_history[start_step:end_step]
    current_pol = recent_policy_history[0]
    
    # Plot background policy spans (Converted to Absolute Seconds)
    for t in range(1, len(recent_policy_history)):
        if recent_policy_history[t] != current_pol or t == len(recent_policy_history) - 1:
            c = policy_colors[current_pol]
            
            # Map indices to absolute time
            start_sec = (start_idx + start_step) / 50.0
            end_sec = (t + start_step) / 50.0
            
            ax.axvspan(start_sec, end_sec, facecolor=c, alpha=0.6)
            
            start_idx = t
            current_pol = recent_policy_history[t]

    i = controller.env_changes.index(env_change)
    prev_env = 'None' if i == 0 else controller.env_changes[i-1][1].replace("Go2Stroll", "")
    env_name = env_name.replace("Go2Stroll", "")
    
    # Absolute time for physical domain change
    ax.axvline(x=last_env_change / 50.0, color='green', linestyle='--', linewidth=2, zorder=5)
    
    # 3. Draw vertical lines for drift detection (Converted to Absolute Seconds)
    active_drift_indices = [idx for idx in controller.drift_indices if start_step < idx < end_step]
    for drift_idx in active_drift_indices:
        drift_time = drift_idx / 50.0
        ax.axvline(x=drift_time, color='red', linestyle='--', linewidth=2, zorder=5)
        
        # --- DRAW MICRO-ROLLOUT BARS ---
        # 20 timesteps = 0.4 seconds per rollout
        for m_idx in range(1, 6):
            rollout_end_time = drift_time + m_idx * 0.4
            end_step_time = end_step / 50.0
            if rollout_end_time < end_step_time:
                ax.axvline(x=rollout_end_time, color='black', linestyle=':', linewidth=2.5, zorder=5)

    # 4. Formatting
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(feet_names, fontweight='bold')
    ax.set_xlabel("Absolute Time (Seconds)", fontsize=12)
    ax.set_title(f"Gait Contact Pattern During Online Adaptation: {prev_env} -> {env_name}", fontsize=14)
    
    # Enforce strict x-limits so there are no empty white gaps at the ends
    ax.set_xlim(start_step / 50.0, end_step / 50.0)
    
    # --- LATEX FORMATTER FOR THESIS NOMENCLATURE ---
    def format_thesis_label(raw_name):
        def clean_part(part):
            part_lower = part.lower()
            if "flat" in part_lower:
                return "flat"
            elif "rough" in part_lower:
                return "rough"
            elif "slippery" in part_lower:
                return "slippery"
            elif "blocked" in part_lower or "broken" in part_lower or "knee" in part_lower:
                return "blocked"
            return part

        if "_AdaptedFrom_" in raw_name:
            parts = raw_name.split("_AdaptedFrom_")
            target = clean_part(parts[0])
            original = clean_part(parts[1])
            return f"$\\pi_{{{target}, {original}}}$"
        else:
            target = clean_part(raw_name)
            return f"$\\pi_{{{target}}}$"

    # Create a clean legend for the background colors with LaTeX formatting
    recent_pols = list(set(recent_policy_history))
    legend_elements = [
        Patch(facecolor=policy_colors[pol], alpha=0.6, label=format_thesis_label(pol)) 
        for pol in recent_pols
    ]
    
    # Add status lines to the legend
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Drift Detected'))
    legend_elements.append(Line2D([0], [0], color='black', linestyle=':', linewidth=2.5, label='Micro-rollout End'))
    legend_elements.append(Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Domain Change'))
    
    # Place legend outside the plot with enlarged font properties
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), 
              title="Legend", title_fontsize=13, fontsize=12)

    plt.tight_layout()
    plt.show()


def plotGPSearch(controller, gp_states=None):
    # Get the sequence of iterations for the most recent drift
    if gp_states is None:
        gp_states = controller.gp_states[-1]

    num_iterations = len(gp_states)
    
    # 1. Collapse the 4D embeddings to a fixed 1D axis using UMAP
    # UMAP preserves both local clustering (redundant policies) AND global distances (domains)
    pol_names_list = list(controller.policy_embeddings.keys())
    emb_matrix = np.array([controller.policy_embeddings[name] for name in pol_names_list])
    
    # n_neighbors dictates how much local vs global structure to preserve. 
    # For a small catalog, a small number (e.g., 3 to 5) works perfectly.
    n_neighbors = min(5, len(pol_names_list) - 1)
    
    reducer = umap.UMAP(n_components=1, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    coords_1d = reducer.fit_transform(emb_matrix).flatten()
    
    x_coords = {name: coord for name, coord in zip(pol_names_list, coords_1d)}
    
    # Extract unique domains and create a Colormap
    base_names = [name.split("_AdaptedFrom_")[0] for name in pol_names_list]
    unique_base_names = list(set(base_names))
    cmap = cm.get_cmap('tab10', len(unique_base_names))
    domain_colors = {domain: cmap(i) for i, domain in enumerate(unique_base_names)}
    present_names = []
    
    # Create a subplot grid: N rows, 1 column. 
    fig, axes = plt.subplots(num_iterations, 1, figsize=(12, 3.5 * num_iterations), 
                             sharex=True, layout="constrained")
    
    if num_iterations == 1: axes = [axes]
    else: axes = axes.flatten()

    for ax_idx, (iteration, base_policy_name, chosen_policy_name, polInfo) in enumerate(gp_states):
        ax = axes[ax_idx]

        xs, means, stds, names = [], [], [], []
        colors, markers, mecs, mews, sizes, zorders = [], [],[], [], [],[]

        # 2. Query GP beliefs for all policies in catalog
        for pol_name in pol_names_list:
            x_val = x_coords[pol_name]
            
            # Find the belief stored in this specific iteration's polInfo
            gp_data = [(m, s) for _, m, s, name in polInfo if pol_name == name]
            if len(gp_data) == 0:
                continue
            
            mean, std = gp_data[0]
            
            xs.append(float(x_val))
            means.append(float(mean[0]))
            stds.append(float(std[0]))
            names.append(pol_name)
            
            # Color based on Target Domain
            base_domain = pol_name.split("_AdaptedFrom_")[0]
            present_names.append(base_domain)
            colors.append(domain_colors[base_domain])
            
            # Shape/Border based on GP Status
            if pol_name == base_policy_name:
                markers.append('D') # Diamond
                mecs.append('black'); mews.append(2.5); sizes.append(100); zorders.append(5)
            elif pol_name == chosen_policy_name:
                markers.append('*') # Star
                mecs.append('red'); mews.append(2.0); sizes.append(300); zorders.append(6)
            else:
                markers.append('o') # Circle
                mecs.append('gray'); mews.append(1.0); sizes.append(60); zorders.append(3)

        # Sort everything by the 1D X-coordinate
        sorted_indices = np.argsort(xs)
        xs, means, stds = np.array(xs)[sorted_indices], np.array(means)[sorted_indices], np.array(stds)[sorted_indices]
        names = [names[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        markers = [markers[i] for i in sorted_indices]
        mecs = [mecs[i] for i in sorted_indices]
        mews = [mews[i] for i in sorted_indices]
        sizes =[sizes[i] for i in sorted_indices]
        zorders = [zorders[i] for i in sorted_indices]

        # 3. Plot the GP's uncertainty bound (Mean ± Std)
        ax.plot(xs, means, color='blue', alpha=0.4, linestyle='--', zorder=1)
        ax.fill_between(xs, means - stds, means + stds, color='blue', alpha=0.1, zorder=0)

        # Scatter the policies individually so we can apply specific markers
        for i in range(len(xs)):
            ax.errorbar(xs[i], means[i], yerr=stds[i], fmt=markers[i], 
                        ecolor='gray', elinewidth=1.5, capsize=3, # Error bar style
                        markerfacecolor=colors[i], markeredgecolor=mecs[i], 
                        markeredgewidth=mews[i], markersize=np.sqrt(sizes[i]), zorder=zorders[i])
            
            # Annotate only the Base and Chosen policies to prevent massive text overlap
            if markers[i] in ['D', '*']:
                # FIX: Push labels strictly UP using fixed screen points (staggered 25 or 45 points high)
                y_offset = 10 if i % 2 == 0 else 45
                
                ax.annotate(names[i], (xs[i], means[i]), xytext=(0, y_offset), 
                            textcoords='offset points', ha='center', fontsize=8, fontweight='bold',
                            # FIX: Set alpha to 0.4 so the box is highly transparent
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=mecs[i], alpha=0.4), zorder=10)

        # Formatting
        ax.set_title(f"Iteration {iteration}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add a little extra top margin so the labels don't get clipped by the subplot boundary
        ax.margins(y=0.2)

    axes[-1].set_xlabel("1D UMAP Projection of Latent Policy Space", fontsize=12)
    fig.supylabel("Predicted Error (GP Mean)", fontsize=14)

    # --- CUSTOM LEGENDS ---
    # Legend 1: Domains (Colors)
    domain_lines =[Line2D([0], [0], marker='o', color='w', markerfacecolor=domain_colors[d], 
                           markersize=10, markeredgecolor='gray', label=d) for d in list(set(present_names))]
    
    # Legend 2: Status (Shapes)
    status_lines = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='black', markeredgewidth=2.5, markersize=8, label='Sampled Policy'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2, markersize=14, label='Chosen (UCB)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1, markersize=8, label='Other')
    ]
    
    blank_line = Line2D([0], [0], color='w', label=' ')

    axes[0].legend(handles=domain_lines + [blank_line] + status_lines, 
                   loc='upper left', bbox_to_anchor=(1.02, 1.05), fontsize=12, framealpha=0.9)

    plt.show()
def plotGPSearchHorizontal(controller, gp_states=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from matplotlib.lines import Line2D
    import umap
    import warnings

    # Get the sequence of iterations for the most recent drift
    if gp_states is None:
        gp_states = controller.gp_states[-1]

    num_iterations = len(gp_states)
    
    # 1. Collapse the 4D embeddings to a fixed 1D axis using UMAP
    pol_names_list = list(controller.policy_embeddings.keys())
    emb_matrix = np.array([controller.policy_embeddings[name] for name in pol_names_list])
    
    n_neighbors = min(5, len(pol_names_list) - 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_components=1, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        coords_1d = reducer.fit_transform(emb_matrix).flatten()
    
    x_coords = {name: coord for name, coord in zip(pol_names_list, coords_1d)}
    
    # Extract unique domains and create a Colormap
    base_names = [name.split("_AdaptedFrom_")[0] for name in pol_names_list]
    unique_base_names = list(set(base_names))
    cmap = cm.get_cmap('tab10', len(unique_base_names))
    domain_colors = {domain: cmap(i) for i, domain in enumerate(unique_base_names)}
    present_names = []
    
    # Create a subplot grid: 1 Row, N Columns. 
    fig, axes = plt.subplots(1, num_iterations, figsize=(3.0 * num_iterations, 5.5), 
                             sharex=True, sharey=True, 
                             gridspec_kw={'wspace': 0.05}, 
                             layout="constrained")
    
    if num_iterations == 1: axes = [axes]
    else: axes = axes.flatten()

    # --- LATEX FORMATTER FOR THESIS NOMENCLATURE ---
    def format_thesis_label(raw_name):
        def clean_part(part):
            part_lower = part.lower()
            if "flat" in part_lower:
                return "flat"
            elif "rough" in part_lower:
                return "rough"
            elif "slippery" in part_lower:
                return "slippery"
            elif "blocked" in part_lower or "broken" in part_lower or "knee" in part_lower:
                return "blocked"
            return part

        if "_AdaptedFrom_" in raw_name:
            parts = raw_name.split("_AdaptedFrom_")
            target = clean_part(parts[0])
            original = clean_part(parts[1])
            return f"$\\pi_{{{target}, {original}}}$"
        else:
            target = clean_part(raw_name)
            return f"$\\pi_{{{target}}}$"

    for ax_idx, (iteration, base_policy_name, chosen_policy_name, polInfo) in enumerate(gp_states):
        ax = axes[ax_idx]

        xs, means, stds, names = [], [], [],[]
        colors, markers, mecs, mews, sizes, zorders = [], [],[], [], [],[]

        # 2. Query GP beliefs for all policies in catalog
        for pol_name in pol_names_list:
            x_val = x_coords[pol_name]
            
            gp_data = [(m, s) for _, m, s, name in polInfo if pol_name == name]
            if len(gp_data) == 0:
                continue
            
            mean, std = gp_data[0]
            base_domain = pol_name.split("_AdaptedFrom_")[0]
            present_names.append(base_domain)
            
            # --- SHAPE/BORDER/OVERLAY LOGIC ---
            # If the base policy is also the chosen one, add both markers on top of each other!
            if pol_name == base_policy_name and pol_name == chosen_policy_name:
                # Add Diamond (Sampled Policy)
                xs.append(float(x_val)); means.append(float(mean[0])); stds.append(float(std[0])); names.append(pol_name); colors.append(domain_colors[base_domain])
                markers.append('D'); mecs.append('black'); mews.append(2.5); sizes.append(100); zorders.append(5)
                
                # Add Star (Chosen Policy)
                xs.append(float(x_val)); means.append(float(mean[0])); stds.append(float(std[0])); names.append(pol_name); colors.append(domain_colors[base_domain])
                markers.append('*'); mecs.append('red'); mews.append(2.0); sizes.append(300); zorders.append(6)
            elif pol_name == base_policy_name:
                xs.append(float(x_val)); means.append(float(mean[0])); stds.append(float(std[0])); names.append(pol_name); colors.append(domain_colors[base_domain])
                markers.append('D'); mecs.append('black'); mews.append(2.5); sizes.append(100); zorders.append(5)
            elif pol_name == chosen_policy_name:
                xs.append(float(x_val)); means.append(float(mean[0])); stds.append(float(std[0])); names.append(pol_name); colors.append(domain_colors[base_domain])
                markers.append('*'); mecs.append('red'); mews.append(2.0); sizes.append(300); zorders.append(6)
            else:
                xs.append(float(x_val)); means.append(float(mean[0])); stds.append(float(std[0])); names.append(pol_name); colors.append(domain_colors[base_domain])
                markers.append('o'); mecs.append('gray'); mews.append(1.0); sizes.append(60); zorders.append(3)

        # Sort everything by the 1D X-coordinate
        sorted_indices = np.argsort(xs)
        xs, means, stds = np.array(xs)[sorted_indices], np.array(means)[sorted_indices], np.array(stds)[sorted_indices]
        names = [names[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        markers = [markers[i] for i in sorted_indices]
        mecs = [mecs[i] for i in sorted_indices]
        mews = [mews[i] for i in sorted_indices]
        sizes = [sizes[i] for i in sorted_indices]
        zorders = [zorders[i] for i in sorted_indices]

        # 3. Plot the GP's uncertainty bound (Mean ± Std)
        ax.plot(xs, means, color='blue', alpha=0.4, linestyle='--', zorder=1)
        ax.fill_between(xs, means - stds, means + stds, color='blue', alpha=0.1, zorder=0)

        # Scatter the policies individually so we can apply specific markers
        annotated_names = set()
        for i in range(len(xs)):
            ax.errorbar(xs[i], means[i], yerr=stds[i], fmt=markers[i], 
                        ecolor='gray', elinewidth=1.5, capsize=3, # Error bar style
                        markerfacecolor=colors[i], markeredgecolor=mecs[i], 
                        markeredgewidth=mews[i], markersize=np.sqrt(sizes[i]), zorder=zorders[i])
            
            # Annotate only the Base and Chosen policies once, preventing overlaps
            if markers[i] in ['D', '*'] and names[i] not in annotated_names:
                annotated_names.add(names[i])
                
                # Stagger label heights cleanly based on unique annotation count
                y_offset = 20 if len(annotated_names) % 2 == 0 else 45
                
                ax.annotate(format_thesis_label(names[i]), (xs[i], means[i]), xytext=(0, y_offset), 
                            textcoords='offset points', ha='center', fontsize=16, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=mecs[i], alpha=0.4), 
                            zorder=10, 
                            in_layout=False)

        # Formatting
        ax.set_title(f"Iteration {iteration}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.margins(x=0.15, y=0.2)

    fig.supxlabel("1D UMAP Projection of Latent Policy Space", fontsize=14)
    fig.supylabel("Predicted Error (GP Mean)", fontsize=14)

    # --- CUSTOM LEGENDS ---
    # Keeps raw environment names in the colormap legend as requested
    domain_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor=domain_colors[d], 
                           markersize=10, markeredgecolor='gray', label=d) for d in list(set(present_names))]
    
    status_lines = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='black', markeredgewidth=2.5, markersize=8, label='Sampled Policy'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2, markersize=14, label='Chosen (UCB)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1, markersize=8, label='Other')
    ]
    
    blank_line = Line2D([0], [0], color='w', label=' ')

    axes[-1].legend(handles=domain_lines + [blank_line] + status_lines, 
                    loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=12, framealpha=0.9)

    plt.show()
    
def statisticDriftHistory(controller, pre_steps=225):
    stat_values = np.array(controller.detector.stat_values)
    p_values    = np.array(controller.detector.p_values)
    drift_indices = controller.drift_indices
    num_drifts = len(drift_indices)

    P_VALUE_THRESHOLD = 1e-4

    # ── Fallback: no drifts ────────────────────────────────────────────────────
    if num_drifts == 0:
        print("No drifts detected to plot.")
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(np.arange(len(stat_values)) / 50.0, stat_values,
                 color='#1f77b4', label="KS statistic")
        ax2 = ax1.twinx()
        ax2.semilogy(np.arange(len(p_values)) / 50.0, p_values,
                     color='#ff7f0e', alpha=0.7, label="p-value")
        ax2.axhline(P_VALUE_THRESHOLD, color='#ff7f0e', linestyle=':', linewidth=1.5)
        ax1.set_xlabel("Absolute Time (Seconds)")
        ax1.set_title("KS-ADWIN history (No drifts)")
        plt.show()
        return

    # ── Main figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, num_drifts,
                             figsize=(4.5 * num_drifts, 4),
                             sharey=True, layout="constrained")
    if num_drifts == 1:
        axes = [axes]

    twin_axes = []
    plotted_env_change = False

    for i, drift_idx in enumerate(drift_indices):
        ax1 = axes[i]

        # ── Slice bounds ───────────────────────────────────────────────────────
        for change_step, env_name in controller.env_changes[::-1]:
            if change_step < drift_idx:
                change_idx = change_step
                break
        start_idx = max(0, change_idx - pre_steps)
        end_idx   = drift_idx + 10

        x_abs = np.arange(start_idx, end_idx) / 50.0
        y_ks  = stat_values[start_idx:end_idx]
        y_p   = p_values[start_idx:end_idx]

        # ── Left axis: KS statistic ────────────────────────────────────────────
        ax1.plot(x_abs, y_ks, color='#1f77b4', linewidth=2,
                 label='KS statistic' if i == 0 else "")
        ax1.fill_between(x_abs, y_ks, color='#1f77b4', alpha=0.15)

        # FIX 1: sharey hides labels on non-first axes — re-enable explicitly
        ax1.tick_params(axis='y', labelleft=True, labelcolor='#1f77b4')

        # ── Right axis: p-values (log scale) ──────────────────────────────────
        ax2 = ax1.twinx()
        twin_axes.append(ax2)

        ax2.fill_between(x_abs, y_p, P_VALUE_THRESHOLD,
                         where=(y_p < P_VALUE_THRESHOLD),
                         color='#ff7f0e', alpha=0.25, interpolate=True)
        ax2.semilogy(x_abs, y_p, color='#ff7f0e', linewidth=1.5, alpha=0.85,
                     label='p-value' if i == 0 else "")
        ax2.axhline(P_VALUE_THRESHOLD, color='#ff7f0e', linestyle=':',
                    linewidth=1.5, alpha=0.9,
                    label=f'Alert threshold (p={P_VALUE_THRESHOLD:.0e})' if i == 0 else "")

        if i == num_drifts - 1:
            ax2.set_ylabel("Policy Performance p-value (log)", fontsize=11,
                           color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')

        # ── Domain change verticals ────────────────────────────────────────────
        for change_step, env_name in controller.env_changes:
            if start_idx <= change_step <= end_idx:
                change_time = change_step / 50.0
                # Draw on ax2 so it sits above the orange fill
                ax2.axvline(change_time, color='green', linestyle=':', linewidth=2.5, zorder=4)
                env_name = env_name.replace("Go2Stroll", "")
                ax1.text(change_time - 0.05, 0.95, f" {env_name}",
                         color='green', rotation=90, va='top', ha='right',
                         fontsize=9, fontweight='bold', alpha=0.8,
                         transform=ax1.get_xaxis_transform(), zorder=5)
                plotted_env_change = True

        # FIX 2: draw red line on ax2 (the top layer) so the orange fill can't bury it
        drift_time = drift_idx / 50.0
        ax2.axvline(drift_time, color='red', linestyle='--', linewidth=2.5, zorder=5,
                    label='Drift Triggered' if i == 0 else "")

        # ── Formatting ─────────────────────────────────────────────────────────
        ax1.set_title(f"Drift Event {i+1} (t = {drift_time:.2f}s)",
                      fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.margins(x=0)

    # ── Sync all right-axes to the same log-scale limits ──────────────────────
    all_p_mins = [ax.get_ylim()[0] for ax in twin_axes]
    all_p_maxs = [ax.get_ylim()[1] for ax in twin_axes]
    shared_ylim = (min(all_p_mins), max(all_p_maxs))
    for ax in twin_axes:
        ax.set_ylim(shared_ylim)

    # ── Global labels ──────────────────────────────────────────────────────────
    axes[0].set_ylabel("KS Statistic", fontsize=12, color='#1f77b4')
    fig.supxlabel("Absolute Time (Seconds)", fontsize=12)

    # ── Legend (merge left + right axis handles from first subplot) ───────────
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = twin_axes[0].get_legend_handles_labels()
    if plotted_env_change:
        h1.append(Line2D([0], [0], color='green', linestyle=':', linewidth=2.5))
        l1.append('Physical Domain Change')
    all_handles = h1 + h2
    all_labels  = l1 + l2
    if all_handles:
        axes[-1].legend(all_handles, all_labels,
                        loc='upper left', bbox_to_anchor=(1.12, 1.0),
                        fontsize=12, framealpha=0.9)
    plt.show()