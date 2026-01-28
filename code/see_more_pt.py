"""
Visualization Script: Compare Two PINN Models (A vs B)

Usage:
    python compare_two_checkpoints.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# Import your classes
from sequential_pinn_log import Config, SequentialPINNTrainer 

# Safety for unpickling
torch.serialization.add_safe_globals([Config])

# ==========================================
# CONFIGURATION
# ==========================================
# Define the two models you want to compare
MODEL_A_PATH = '/Users/xiaoxizhou/Downloads/su_26/adrian_surf/outputs_sequential_log/best_model.pt'       # e.g., Best Validation Loss
MODEL_B_PATH = '/Users/xiaoxizhou/Downloads/su_26/adrian_surf/outputs_sequential_log/sequential_pinn_log.pt'  # e.g., End of training (or a specific epoch)

DATA_PATH = '/Users/xiaoxizhou/Downloads/su_26/adrian_surf/reaction_data_log.npz'
OUTPUT_DIR = 'outputs_comparison'

SPECIES_NAMES = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']
COLORS = ['r', 'g', 'k', 'm', 'c', 'orange', 'purple', 'pink', 'brown']

# ==========================================
# Helper Functions
# ==========================================
def load_data_and_setup():
    """Load the ground truth data once."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Device: {device}")
    
    # Load Data
    data = np.load(DATA_PATH)
    inputs = data['inputs']
    
    # Reconstruct Time and Initial State
    initial_state = inputs[0, :11]
    log_dt_array = inputs[:, 11]
    dt_array = 10 ** log_dt_array
    
    time_array = np.zeros(len(dt_array) + 1)
    for i in range(len(dt_array)):
        time_array[i+1] = time_array[i] + dt_array[i]
        
    return device, initial_state, log_dt_array, time_array

def run_model_rollout(model_path, device, initial_state, log_dt_array, label):
    """Load a specific model and run the trajectory."""
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Skipping.")
        return None

    print(f"\nLoading {label}: {model_path}")
    trainer = SequentialPINNTrainer(config=Config, device=device)
    trainer.load_model(model_path)
    
    trajectory = [initial_state]
    current_state = initial_state.copy()
    
    print(f"  Running rollout ({len(log_dt_array)} steps)...")
    for log_dt in log_dt_array:
        next_state = trainer.predict_single_step(current_state, log_dt)
        trajectory.append(next_state)
        current_state = next_state
        
    return np.array(trajectory)

# ==========================================
# Plotting
# ==========================================
def plot_model_comparison(time, traj_A, traj_B, species_names, colors):
    """Plots Model A vs Model B for all species."""
    
    if traj_A is None or traj_B is None:
        print("One of the models failed to load. Cannot plot comparison.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Species Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, name in enumerate(species_names):
        c = colors[i % len(colors)]
        
        # Model A = Solid Line
        ax.semilogx(time, traj_A[:, i], color=c, linestyle='-', linewidth=2, alpha=0.8, 
                   label=f'{name}' if i==0 else "") # Label trick handled below
        
        # Model B = Dotted Line
        ax.semilogx(time, traj_B[:, i], color=c, linestyle=':', linewidth=3, alpha=0.9)

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=2, linestyle='-'),
                    Line2D([0], [0], color='black', lw=2, linestyle=':')]
    
    ax.legend(custom_lines, ['Model A (Solid)', 'Model B (Dotted)'], loc='upper left')
    
    # Add species labels (optional, or rely on color coding from previous plots)
    # For clarity, let's just title it
    ax.set_title(f'Model Comparison\nA: {os.path.basename(MODEL_A_PATH)} vs \nB: {os.path.basename(MODEL_B_PATH)}', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('log(Mass Fraction)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time[1], time[-1]])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/model_comparison_species.png', dpi=200)
    print(f"Saved species comparison to {OUTPUT_DIR}/model_comparison_species.png")
    plt.show()

    # 2. Temperature Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Temp is index 10
    T_A = 10 ** traj_A[:, 10]
    T_B = 10 ** traj_B[:, 10]
    
    ax2.semilogx(time, T_A, 'b-', linewidth=2, label='Model A')
    ax2.semilogx(time, T_B, 'r:', linewidth=3, label='Model B')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Temperature Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([time[1], time[-1]])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/model_comparison_temp.png', dpi=200)
    print(f"Saved temp comparison to {OUTPUT_DIR}/model_comparison_temp.png")
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    device, initial_state, log_dt, time = load_data_and_setup()
    
    # Run Rollouts
    traj_A = run_model_rollout(MODEL_A_PATH, device, initial_state, log_dt, "Model A")
    traj_B = run_model_rollout(MODEL_B_PATH, device, initial_state, log_dt, "Model B")
    
    # Plot
    plot_model_comparison(time, traj_A, traj_B, SPECIES_NAMES, COLORS)