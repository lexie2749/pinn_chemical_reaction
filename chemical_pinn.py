"""
Physics-Informed Neural Network for Chemical Reaction Modeling
Based on: "Physics informed neural network framework for unsteady discretized reduced order system"
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ChemicalPINN(nn.Module):
    """
    Physics-Informed Neural Network for chemical reactions
    Following the ANN-DisPINN architecture from the paper
    """
    def __init__(self, input_dim=12, output_dim=11, hidden_layers=[124, 64, 24, 8]):
        super(ChemicalPINN, self).__init__()
        
        # Build the network architecture (4 hidden layers as in paper)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Activation function
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class PhysicsLoss:
    """
    Physics-based loss terms for chemical reactions
    Implements conservation laws as soft constraints
    """
    def __init__(self, species_names, molar_masses, atomic_composition):
        """
        Args:
            species_names: List of species names
            molar_masses: Dict of molar masses (kg/mol)
            atomic_composition: Dict mapping species to atom counts
                                e.g., {'CO2': {'C': 1, 'O': 2}, ...}
        """
        self.species_names = species_names
        self.n_species = len(species_names)
        self.molar_masses = molar_masses
        self.atomic_composition = atomic_composition
        
        # Gas constant (J/(mol·K))
        self.R = 8.314
    
    def atom_conservation_loss(self, X_input, X_output, rho_input, rho_output, T_input, T_output):
        """
        Conservation of atoms (C, O, N, Ar)
        """
        batch_size = X_input.shape[0]
        
        # Calculate mixture molecular weight (kg/mol)
        W_mix_input = torch.zeros(batch_size, device=X_input.device)
        W_mix_output = torch.zeros(batch_size, device=X_output.device)
        
        for i, species in enumerate(self.species_names):
            W_i = self.molar_masses[species]
            W_mix_input += X_input[:, i] * W_i
            W_mix_output += X_output[:, i] * W_i
        
        # --- FIX START ---
        # Squeeze rho to shape (batch_size,) to match W_mix shape
        # rho_input was (batch, 1), W_mix_input is (batch,)
        n_total_input = rho_input.view(-1) / W_mix_input
        n_total_output = rho_output.view(-1) / W_mix_output
        # --- FIX END ---
        
        # Count atoms for each element
        atom_names = ['C', 'O', 'N', 'AR']
        
        losses = []
        
        for atom in atom_names:
            # Count atoms at input state
            atom_count_input = torch.zeros(batch_size, device=X_input.device)
            atom_count_output = torch.zeros(batch_size, device=X_output.device)
            
            for i, species in enumerate(self.species_names):
                n_atoms = self.atomic_composition.get(species, {}).get(atom, 0)
                
                # Moles of this species per unit volume
                # Now both tensors are 1D (batch_size,), so multiplication is element-wise
                n_species_input = X_input[:, i] * n_total_input
                n_species_output = X_output[:, i] * n_total_output
                
                # Add atom contribution
                atom_count_input += n_atoms * n_species_input
                atom_count_output += n_atoms * n_species_output
            
            # Conservation: atoms_in = atoms_out
            loss = torch.mean((atom_count_input - atom_count_output)**2)
            losses.append(loss)
        
        return sum(losses) / len(losses)
        
    
    def energy_conservation_loss(self, X_input, X_output, rho_input, rho_output, T_input, T_output):
        """
        Conservation of internal energy (constant volume process)
        
        For constant volume: ΔU = 0 (assuming adiabatic, no work)
        This is a simplification - in reality, temperature changes due to reaction enthalpy
        
        For now, we enforce that the total enthalpy is conserved
        """
        # Simplified: We assume small temperature changes
        # A more rigorous approach would require thermodynamic data
        
        # For constant volume adiabatic: we can check if energy balance holds
        # This is complex, so we use a simplified constraint:
        # The temperature change should be physically reasonable
        
        T_change = torch.abs(T_output - T_input)
        
        # Penalize very large temperature changes (> 1000 K) as unphysical
        max_reasonable_change = 1000.0
        loss = torch.mean(torch.relu(T_change - max_reasonable_change)**2)
        
        return loss
    
    def volume_conservation_loss(self, rho_input, rho_output):
        """
        For constant volume process with ideal gas:
        Density should remain constant if mass is conserved
        
        Actually, for constant V and constant mass: ρ = constant
        """
        loss = torch.mean((rho_input - rho_output)**2)
        return loss
    
    def sum_of_fractions_loss(self, X_output):
        """
        Molar fractions must sum to 1
        Σ X_i = 1
        """
        sum_X = torch.sum(X_output[:, :self.n_species], dim=1)
        loss = torch.mean((sum_X - 1.0)**2)
        return loss
    
    def non_negativity_loss(self, X_output):
        """
        Molar fractions must be non-negative
        X_i >= 0
        """
        # Penalize negative values
        loss = torch.mean(torch.relu(-X_output[:, :self.n_species])**2)
        return loss


class ChemicalReactionPINN:
    """
    Complete PINN training framework for chemical reactions
    """
    def __init__(self, species_names, molar_masses, atomic_composition,
                 hidden_layers=[124, 64, 24, 8], device='cpu'):
        
        self.device = device
        self.species_names = species_names
        self.n_species = len(species_names)
        
        # Create network
        input_dim = self.n_species + 3  # [X1,...,X9, rho, T, dt]
        output_dim = self.n_species + 2  # [X1,...,X9, rho, T]
        
        self.model = ChemicalPINN(input_dim, output_dim, hidden_layers).to(device)
        
        # Create physics loss calculator
        self.physics_loss = PhysicsLoss(species_names, molar_masses, atomic_composition)
        
        # Loss weights
        self.lambda_data = 1.0
        self.lambda_atom = 1.0
        self.lambda_energy = 0.1
        self.lambda_volume = 1.0
        self.lambda_sum = 10.0
        self.lambda_nonneg = 5.0
        
        # Optimizer
        self.optimizer = None
        self.scheduler = None
        
        # Loss history
        self.loss_history = {
            'total': [],
            'data': [],
            'physics': [],
            'atom': [],
            'energy': [],
            'volume': [],
            'sum': [],
            'nonneg': []
        }
    
    def data_loss(self, y_pred, y_true):
        """Mean squared error for data-driven term"""
        return torch.mean((y_pred - y_true)**2)
    
    def compute_loss(self, inputs, outputs_true):
        """
        Compute total loss: L = L_data + λ * L_physics
        
        Args:
            inputs: [X_in (9), rho_in, T_in, dt] - shape (batch, 12)
            outputs_true: [X_out (9), rho_out, T_out] - shape (batch, 11)
        """
        # Forward pass
        outputs_pred = self.model(inputs)
        
        # Extract components
        X_input = inputs[:, :self.n_species]
        rho_input = inputs[:, self.n_species:self.n_species+1]
        T_input = inputs[:, self.n_species+1:self.n_species+2]
        
        X_pred = outputs_pred[:, :self.n_species]
        rho_pred = outputs_pred[:, self.n_species:self.n_species+1]
        T_pred = outputs_pred[:, self.n_species+1:self.n_species+2]
        
        X_true = outputs_true[:, :self.n_species]
        rho_true = outputs_true[:, self.n_species:self.n_species+1]
        T_true = outputs_true[:, self.n_species+1:self.n_species+2]
        
        # Data-driven loss
        loss_data = self.data_loss(outputs_pred, outputs_true)
        
        # Physics-based losses
        loss_atom = self.physics_loss.atom_conservation_loss(
            X_input, X_pred, rho_input, rho_pred, T_input, T_pred
        )
        
        loss_energy = self.physics_loss.energy_conservation_loss(
            X_input, X_pred, rho_input, rho_pred, T_input, T_pred
        )
        
        loss_volume = self.physics_loss.volume_conservation_loss(
            rho_input, rho_pred
        )
        
        loss_sum = self.physics_loss.sum_of_fractions_loss(X_pred)
        loss_nonneg = self.physics_loss.non_negativity_loss(X_pred)
        
        # Total physics loss
        loss_physics = (self.lambda_atom * loss_atom +
                       self.lambda_energy * loss_energy +
                       self.lambda_volume * loss_volume +
                       self.lambda_sum * loss_sum +
                       self.lambda_nonneg * loss_nonneg)
        
        # Total loss
        loss_total = self.lambda_data * loss_data + loss_physics
        
        # Store individual losses
        losses = {
            'total': loss_total.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'atom': loss_atom.item(),
            'energy': loss_energy.item(),
            'volume': loss_volume.item(),
            'sum': loss_sum.item(),
            'nonneg': loss_nonneg.item()
        }
        
        return loss_total, losses
    
    def train(self, train_loader, epochs=5000, learning_rate=0.001, 
              print_every=500, val_loader=None):
        """
        Train the PINN
        
        Args:
            train_loader: DataLoader with training data
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            print_every: Print frequency
            val_loader: Optional validation DataLoader
        """
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500
        )
        
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {key: 0.0 for key in self.loss_history.keys()}
            n_batches = 0
            
            for inputs, outputs in train_loader:
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute loss
                loss, losses = self.compute_loss(inputs, outputs)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value
                n_batches += 1
            
            # Average losses
            for key in epoch_losses.keys():
                epoch_losses[key] /= n_batches
                self.loss_history[key].append(epoch_losses[key])
            
            # Update learning rate
            self.scheduler.step(epoch_losses['total'])
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.1f}s")
                print(f"  Total Loss: {epoch_losses['total']:.6e}")
                print(f"  Data Loss:  {epoch_losses['data']:.6e}")
                print(f"  Physics:    {epoch_losses['physics']:.6e}")
                print(f"    - Atom:   {epoch_losses['atom']:.6e}")
                print(f"    - Volume: {epoch_losses['volume']:.6e}")
                print(f"    - Sum:    {epoch_losses['sum']:.6e}")
                
                # Validation
                if val_loader is not None:
                    val_loss = self.validate(val_loader)
                    print(f"  Val Loss:   {val_loss:.6e}")
                print("-" * 80)
        
        print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    
    def validate(self, val_loader):
        """Compute validation loss"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                
                loss, _ = self.compute_loss(inputs, outputs)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def predict(self, inputs):
        """
        Make predictions
        
        Args:
            inputs: numpy array or tensor of shape (n_samples, 12)
        
        Returns:
            predictions: numpy array of shape (n_samples, 11)
        """
        self.model.eval()
        
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return outputs.cpu().numpy()
    
    def plot_training_history(self, save_path=None):
        """Plot training loss curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total and data loss
        ax = axes[0, 0]
        ax.semilogy(self.loss_history['total'], label='Total Loss', linewidth=2)
        ax.semilogy(self.loss_history['data'], label='Data Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total and Data Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Physics loss
        ax = axes[0, 1]
        ax.semilogy(self.loss_history['physics'], label='Total Physics', linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Physics Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Physics components
        ax = axes[1, 0]
        ax.semilogy(self.loss_history['atom'], label='Atom Conservation', linewidth=2)
        ax.semilogy(self.loss_history['volume'], label='Volume Conservation', linewidth=2)
        ax.semilogy(self.loss_history['sum'], label='Sum of Fractions', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Physics Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy and non-negativity
        ax = axes[1, 1]
        ax.semilogy(self.loss_history['energy'], label='Energy', linewidth=2)
        ax.semilogy(self.loss_history['nonneg'], label='Non-negativity', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Additional Constraints')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Data Loading and Preprocessing

def load_and_preprocess_data(data_path='reaction_data.npz', train_split=0.8):
    """
    Load data from npz file and prepare for training
    
    Args:
        data_path: Path to the .npz file
        train_split: Fraction of data for training
    
    Returns:
        train_loader, val_loader, normalization_params
    """
    # Load data
    data = np.load(data_path)
    inputs = data['inputs']  # Shape: (n_samples, 12)
    outputs = data['outputs']  # Shape: (n_samples, 11)
    
    print(f"Loaded data shapes:")
    print(f"  Inputs: {inputs.shape}")
    print(f"  Outputs: {outputs.shape}")
    
    # Normalize data
    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0) + 1e-8
    output_mean = outputs.mean(axis=0)
    output_std = outputs.std(axis=0) + 1e-8
    
    inputs_normalized = (inputs - input_mean) / input_std
    outputs_normalized = (outputs - output_mean) / output_std
    
    # Split into train/val
    n_samples = inputs.shape[0]
    n_train = int(n_samples * train_split)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(inputs_normalized[train_indices]),
        torch.FloatTensor(outputs_normalized[train_indices])
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(inputs_normalized[val_indices]),
        torch.FloatTensor(outputs_normalized[val_indices])
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    normalization_params = {
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std
    }
    
    print(f"\nData split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, normalization_params, inputs, outputs


# Main Training

if __name__ == "__main__":
    # Species information
    species_names = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']
    
    # Molar masses (kg/mol)
    molar_masses = {
        'CO2': 0.04401,
        'O2': 0.032,
        'N2': 0.028014,
        'CO': 0.02801,
        'NO': 0.03001,
        'C': 0.01201,
        'O': 0.016,
        'N': 0.014007,
        'AR': 0.039948
    }
    
    # Atomic composition
    atomic_composition = {
        'CO2': {'C': 1, 'O': 2},
        'O2': {'O': 2},
        'N2': {'N': 2},
        'CO': {'C': 1, 'O': 1},
        'NO': {'N': 1, 'O': 1},
        'C': {'C': 1},
        'O': {'O': 1},
        'N': {'N': 1},
        'AR': {'AR': 1}
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, norm_params, inputs_raw, outputs_raw = \
        load_and_preprocess_data('reaction_data.npz', train_split=0.8)
    
    # Create PINN
    pinn = ChemicalReactionPINN(
        species_names=species_names,
        molar_masses=molar_masses,
        atomic_composition=atomic_composition,
        hidden_layers=[124, 64, 24, 8],
        device=device
    )
    
    # Train (reduced epochs for demo)
    pinn.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1000,  # Reduced for demo
        learning_rate=0.001,
        print_every=200
    )
    
    # Plot training history
    pinn.plot_training_history(save_path='/Users/xiaoxizhou/Downloads/surf/adrian_surf/training_history.png')
    
    # Save model
    torch.save({
        'model_state_dict': pinn.model.state_dict(),
        'normalization_params': norm_params,
        'species_names': species_names
    }, '/Users/xiaoxizhou/Downloads/surf/adrian_surf/training_history.png')
    
    print("\nTraining complete! Model saved.")
