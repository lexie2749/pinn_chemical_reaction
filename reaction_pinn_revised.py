"""
Reaction PINN (Physics-Informed Neural Network) for Chemical Kinetics
Revised version with improved structure, bug fixes, and additional features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from pathlib import Path


# ==========================================
# Configuration
# ==========================================
class Config:
    """Centralized configuration for the PINN model."""
    # Model architecture
    INPUT_DIM = 12   # [Species(9), rho, T, dt]
    OUTPUT_DIM = 11  # [Species(9), rho, T]
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    
    # Normalization scales
    SCALE_T = 8000.0
    SCALE_RHO = 0.002
    SCALE_DT = 1e-5
    
    # Training parameters
    EPOCHS = 2000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = None  # None = full batch
    PRINT_EVERY = 500
    
    # Paths
    MODEL_PATH = "pinn_model.pt"
    OUTPUT_DIR = "outputs"
    
    # Species names
    SPECIES_NAMES = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']


# ==========================================
# Model Definition
# ==========================================
class ReactionPINN(nn.Module):
    """Physics-Informed Neural Network for chemical reaction modeling."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)
    
    def predict_with_softmax(self, x):
        """Forward pass with softmax applied to species outputs."""
        out = self.forward(x)
        X_pred = torch.nn.functional.softmax(out[:, :9], dim=1)
        return torch.cat([X_pred, out[:, 9:]], dim=1)


# ==========================================
# Data Loading and Preprocessing
# ==========================================
def generate_dummy_data(n_samples=1000):
    """Generate synthetic data for demonstration purposes."""
    print("⚠️  Warning: Generating dummy data for demonstration...")
    
    t = np.linspace(0, 0.01, n_samples)
    dt = np.diff(t, prepend=0)
    dt[0] = dt[1]
    
    # Construct synthetic species data
    species = np.abs(np.sin(t.reshape(-1, 1) * np.arange(1, 10)))
    species = species / species.sum(axis=1, keepdims=True)  # Normalize to sum=1
    
    rho = 1.0 - 0.1 * t
    T = 2000.0 + 500.0 * t
    
    inputs = np.column_stack([species[:-1], rho[:-1], T[:-1], dt[:-1]])
    outputs = np.column_stack([species[1:], rho[1:], T[1:]])
    
    return inputs, outputs, dt[:-1]


def load_data(data_path='reaction_data.npz'):
    """Load data from file or generate dummy data."""
    print("📂 Loading data...")
    
    if os.path.exists(data_path):
        data = np.load(data_path)
        inputs = data['inputs']
        outputs = data['outputs']
        dt_array = data['dt']
        print(f"   Loaded {len(inputs)} samples from {data_path}")
    else:
        inputs, outputs, dt_array = generate_dummy_data()
        print(f"   Generated {len(inputs)} synthetic samples")
    
    return inputs, outputs, dt_array


def normalize_data(inputs, outputs, config=Config):
    """Apply normalization to input and output tensors."""
    inputs_norm = inputs.clone()
    outputs_norm = outputs.clone()
    
    # Normalize inputs: [X(9), rho, T, dt]
    inputs_norm[:, 9] /= config.SCALE_RHO
    inputs_norm[:, 10] /= config.SCALE_T
    inputs_norm[:, 11] /= config.SCALE_DT
    
    # Normalize outputs: [X(9), rho, T]
    outputs_norm[:, 9] /= config.SCALE_RHO
    outputs_norm[:, 10] /= config.SCALE_T
    
    return inputs_norm, outputs_norm


# ==========================================
# Training
# ==========================================
def train_model(model, inputs_norm, targets_norm, config=Config, device='cpu'):
    """Train the PINN model."""
    print(f"\n🚀 Starting Training for {config.EPOCHS} epochs...")
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)
    loss_fn = nn.MSELoss()
    
    loss_history = []
    best_loss = float('inf')
    
    for epoch in range(config.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with physics constraint (softmax for species)
        pred = model.predict_with_softmax(inputs_norm)
        
        loss = loss_fn(pred, targets_norm)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # Learning rate scheduling
        scheduler.step(loss_val)
        
        # Track best model
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = model.state_dict().copy()
        
        if epoch % config.PRINT_EVERY == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch:5d} | Loss: {loss_val:.6e} | LR: {lr:.2e}")
    
    # Restore best model
    model.load_state_dict(best_state)
    print(f"✅ Training completed. Best loss: {best_loss:.6e}")
    
    return loss_history


def save_model(model, path, config=Config):
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': config.INPUT_DIM,
            'output_dim': config.OUTPUT_DIM,
            'hidden_dim': config.HIDDEN_DIM,
            'num_layers': config.NUM_LAYERS,
        }
    }, path)
    print(f"💾 Model saved to {path}")


def load_model(path, device='cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint['config']
    
    model = ReactionPINN(
        cfg['input_dim'], cfg['output_dim'],
        cfg['hidden_dim'], cfg['num_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"📂 Model loaded from {path}")
    return model


# ==========================================
# Time Integration (Rollout Prediction)
# ==========================================
def rollout_prediction(model, initial_state, dt_sequence, config=Config, device='cpu'):
    """Perform time integration using the trained PINN."""
    print("\n🔄 Starting Time Integration (Rollout)...")
    
    model.eval()
    current_state = initial_state.copy()
    pred_history = [current_state.copy()]
    
    with torch.no_grad():
        for i, dt_val in enumerate(dt_sequence):
            # Prepare normalized input
            inp = torch.zeros(1, config.INPUT_DIM, device=device)
            inp[0, :9] = torch.tensor(current_state[:9], dtype=torch.float32)
            inp[0, 9] = current_state[9] / config.SCALE_RHO
            inp[0, 10] = current_state[10] / config.SCALE_T
            inp[0, 11] = dt_val / config.SCALE_DT
            
            # PINN prediction
            out_norm = model(inp)
            
            # Denormalize outputs
            pred_X = torch.nn.functional.softmax(out_norm[0, :9], dim=0).cpu().numpy()
            pred_rho = out_norm[0, 9].item() * config.SCALE_RHO
            pred_T = out_norm[0, 10].item() * config.SCALE_T
            
            # Update state
            current_state = np.concatenate([pred_X, [pred_rho], [pred_T]])
            pred_history.append(current_state)
    
    print(f"   Completed {len(dt_sequence)} time steps")
    return np.array(pred_history)


# ==========================================
# Visualization
# ==========================================
def plot_training_loss(loss_history, output_dir):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, 'b-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: {path}")


def plot_temperature_comparison(time, gt_history, pred_history, output_dir):
    """Plot temperature evolution comparison."""
    plt.figure(figsize=(10, 6))
    plt.semilogx(time, gt_history[:, 10], 'k-', linewidth=3, alpha=0.6, label='Ground Truth')
    plt.semilogx(time, pred_history[:, 10], 'r--', linewidth=2.5, label='PINN Prediction')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Evolution')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'temperature_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: {path}")


def plot_species_comparison(time, gt_history, pred_history, output_dir, config=Config):
    """Plot species molar fraction evolution comparison."""
    num_species = min(6, len(config.SPECIES_NAMES))
    colors = plt.cm.tab10(np.linspace(0, 1, num_species))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i in range(num_species):
        color = colors[i]
        name = config.SPECIES_NAMES[i]
        
        # Ground Truth (solid line)
        y_gt = np.log10(np.maximum(gt_history[:, i], 1e-20))
        ax.semilogx(time, y_gt, '-', color=color, linewidth=2.5, alpha=0.6)
        
        # PINN Prediction (dashed line)
        y_pred = np.log10(np.maximum(pred_history[:, i], 1e-20))
        ax.semilogx(time, y_pred, '--', color=color, linewidth=2, label=name)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('log₁₀(Mole Fraction)')
    ax.set_title('Species Molar Fraction Evolution')
    ax.grid(True, which="both", alpha=0.3)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2.5, alpha=0.6, label='Ground Truth'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='PINN Prediction'),
    ]
    legend_elements.extend([
        Line2D([0], [0], color=colors[i], linewidth=2, label=config.SPECIES_NAMES[i])
        for i in range(num_species)
    ])
    ax.legend(handles=legend_elements, loc='upper right', ncol=2)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'species_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: {path}")


def plot_density_comparison(time, gt_history, pred_history, output_dir):
    """Plot density evolution comparison."""
    plt.figure(figsize=(10, 6))
    plt.semilogx(time, gt_history[:, 9], 'k-', linewidth=3, alpha=0.6, label='Ground Truth')
    plt.semilogx(time, pred_history[:, 9], 'b--', linewidth=2.5, label='PINN Prediction')
    plt.xlabel('Time (s)')
    plt.ylabel('Density (kg/m³)')
    plt.title('Density Evolution')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'density_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: {path}")


# ==========================================
# Main Pipeline
# ==========================================
def run_pinn_analysis(config=Config):
    """Main function to run the complete PINN analysis pipeline."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Running on device: {device}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    inputs_raw, outputs_raw, dt_array_raw = load_data()
    
    # Convert to tensors
    inputs_tensor = torch.tensor(inputs_raw, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(outputs_raw, dtype=torch.float32, device=device)
    
    # Normalize
    inputs_norm, targets_norm = normalize_data(inputs_tensor, targets_tensor, config)
    
    # Initialize model
    model = ReactionPINN(
        config.INPUT_DIM, config.OUTPUT_DIM,
        config.HIDDEN_DIM, config.NUM_LAYERS
    ).to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    loss_history = train_model(model, inputs_norm, targets_norm, config, device)
    
    # Save model
    save_model(model, os.path.join(config.OUTPUT_DIR, config.MODEL_PATH), config)
    
    # Rollout prediction
    initial_state = inputs_raw[0, :11].copy()
    dt_sequence = dt_array_raw.flatten()
    pred_history = rollout_prediction(model, initial_state, dt_sequence, config, device)
    
    # Prepare ground truth for comparison
    limit = len(dt_sequence) + 1
    gt_history = np.vstack([inputs_raw[0, :11], outputs_raw[:limit-1, :11]])
    time_accumulated = np.concatenate(([0], np.cumsum(dt_sequence)))
    
    # Generate plots
    print("\n📈 Generating plots...")
    plot_training_loss(loss_history, config.OUTPUT_DIR)
    plot_temperature_comparison(time_accumulated, gt_history, pred_history, config.OUTPUT_DIR)
    plot_species_comparison(time_accumulated, gt_history, pred_history, config.OUTPUT_DIR, config)
    plot_density_comparison(time_accumulated, gt_history, pred_history, config.OUTPUT_DIR)
    
    # Compute and report errors
    temp_rmse = np.sqrt(np.mean((gt_history[:, 10] - pred_history[:, 10])**2))
    rho_rmse = np.sqrt(np.mean((gt_history[:, 9] - pred_history[:, 9])**2))
    species_rmse = np.sqrt(np.mean((gt_history[:, :9] - pred_history[:, :9])**2))
    
    print("\n📊 Rollout Error Summary:")
    print(f"   Temperature RMSE: {temp_rmse:.4f} K")
    print(f"   Density RMSE:     {rho_rmse:.6f} kg/m³")
    print(f"   Species RMSE:     {species_rmse:.6e}")
    
    print(f"\n✅ Analysis complete! Results saved to '{config.OUTPUT_DIR}/'")
    
    return model, pred_history, gt_history


if __name__ == "__main__":
    run_pinn_analysis()
