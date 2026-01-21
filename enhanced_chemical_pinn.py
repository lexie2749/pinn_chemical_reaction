"""
Enhanced Chemical PINN with Reaction Kinetics
Incorporates Arrhenius reaction rates from airNASA9ions.yaml mechanism
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

torch.manual_seed(42)
np.random.seed(42)


class ReactionKinetics:
    """
    Simplified reaction kinetics based on airNASA9ions.yaml
    Focuses on major reactions affecting temperature drop
    """
    def __init__(self):
        # Universal gas constant (J/(mol·K))
        self.R = 8.314
        
        # Major dissociation reactions (endothermic - cause temperature drop)
        self.reactions = {
            # Reaction 1: CO2 + M <=> CO + O + M
            'CO2_diss': {
                'A': 6.9e20,           # Pre-exponential factor (cm³/mol/s)
                'b': -1.5,             # Temperature exponent
                'Ea': 1.2575e5 * 4.184,  # Activation energy (cal/mol -> J/mol)
                'reactants': {'CO2': 1},
                'products': {'CO': 1, 'O': 1},
                'delta_H': 532000  # Enthalpy of dissociation (J/mol) - endothermic
            },
            
            # Reaction 5: N2 + M <=> N + N + M
            'N2_diss': {
                'A': 7.0e21,
                'b': -1.6,
                'Ea': 2.24951e5 * 4.184,
                'reactants': {'N2': 1},
                'products': {'N': 2},
                'delta_H': 945000  # J/mol - very endothermic
            },
            
            # Reaction 7: O2 + M <=> O + O + M  
            'O2_diss': {
                'A': 2.0e21,
                'b': -1.5,
                'Ea': 1.1796e5 * 4.184,
                'reactants': {'O2': 1},
                'products': {'O': 2},
                'delta_H': 498000  # J/mol - endothermic
            },
            
            # Reaction 19: N2 + O <=> NO + N (Zeldovich)
            'N2_O': {
                'A': 6.0e13,
                'b': 0.1,
                'Ea': 7.55514e4 * 4.184,
                'reactants': {'N2': 1, 'O': 1},
                'products': {'NO': 1, 'N': 1},
                'delta_H': 315000  # J/mol - endothermic
            },
            
            # Reaction 20: O2 + N <=> NO + O
            'O2_N': {
                'A': 2.49e9,
                'b': 1.18,
                'Ea': 7968.67 * 4.184,
                'reactants': {'O2': 1, 'N': 1},
                'products': {'NO': 1, 'O': 1},
                'delta_H': -32000  # J/mol - slightly exothermic
            }
        }
        
        # Species molar masses (kg/mol)
        self.molar_masses = {
            'CO2': 0.04401, 'O2': 0.032, 'N2': 0.028014,
            'CO': 0.02801, 'NO': 0.03001, 'C': 0.01201,
            'O': 0.016, 'N': 0.014007, 'AR': 0.039948
        }
    
    def arrhenius_rate(self, A, b, Ea, T):
        """
        Calculate Arrhenius reaction rate: k = A * T^b * exp(-Ea/(RT))
        
        Args:
            A: Pre-exponential factor
            b: Temperature exponent  
            Ea: Activation energy (J/mol)
            T: Temperature (K)
        
        Returns:
            k: Rate constant
        """
        return A * torch.pow(T, b) * torch.exp(-Ea / (self.R * T))
    
    def compute_reaction_rates(self, X, T, rho):
        """
        Compute reaction rates for major reactions
        
        Args:
            X: Molar fractions (batch, 9) - [CO2, O2, N2, CO, NO, C, O, N, AR]
            T: Temperature (batch, 1) - Kelvin
            rho: Density (batch, 1) - kg/m³
        
        Returns:
            rates: Dictionary of reaction rates (mol/m³/s)
        """
        batch_size = X.shape[0]
        
        # Species mapping
        species_map = {
            'CO2': 0, 'O2': 1, 'N2': 2, 'CO': 3, 'NO': 4,
            'C': 5, 'O': 6, 'N': 7, 'AR': 8
        }
        
        # Calculate mixture molecular weight
        W_mix = torch.zeros(batch_size, device=X.device)
        for name, idx in species_map.items():
            W_mix += X[:, idx] * self.molar_masses[name]
        
        # Total molar concentration (mol/m³)
        # Convert from kg/m³ to mol/m³: C_total = rho / W_mix
        C_total = rho.squeeze() / W_mix
        
        # Individual species concentrations (mol/m³)
        C = {}
        for name, idx in species_map.items():
            C[name] = X[:, idx] * C_total
        
        rates = {}
        
        # CO2 dissociation
        k_CO2 = self.arrhenius_rate(
            self.reactions['CO2_diss']['A'],
            self.reactions['CO2_diss']['b'],
            self.reactions['CO2_diss']['Ea'],
            T.squeeze()
        )
        # Forward rate (simplified, ignoring reverse)
        rates['CO2_diss'] = k_CO2 * C['CO2'] * C_total  # Third-body reaction
        
        # N2 dissociation  
        k_N2 = self.arrhenius_rate(
            self.reactions['N2_diss']['A'],
            self.reactions['N2_diss']['b'],
            self.reactions['N2_diss']['Ea'],
            T.squeeze()
        )
        rates['N2_diss'] = k_N2 * C['N2'] * C_total
        
        # O2 dissociation
        k_O2 = self.arrhenius_rate(
            self.reactions['O2_diss']['A'],
            self.reactions['O2_diss']['b'],
            self.reactions['O2_diss']['Ea'],
            T.squeeze()
        )
        rates['O2_diss'] = k_O2 * C['O2'] * C_total
        
        # Zeldovich mechanism: N2 + O <=> NO + N
        k_N2_O = self.arrhenius_rate(
            self.reactions['N2_O']['A'],
            self.reactions['N2_O']['b'],
            self.reactions['N2_O']['Ea'],
            T.squeeze()
        )
        rates['N2_O'] = k_N2_O * C['N2'] * C['O']
        
        # O2 + N <=> NO + O
        k_O2_N = self.arrhenius_rate(
            self.reactions['O2_N']['A'],
            self.reactions['O2_N']['b'],
            self.reactions['O2_N']['Ea'],
            T.squeeze()
        )
        rates['O2_N'] = k_O2_N * C['O2'] * C['N']
        
        return rates, C_total
    
    def compute_heat_release(self, rates):
        """
        Compute heat release rate from reactions (J/m³/s)
        Negative = heat consumed (endothermic, temperature drops)
        Positive = heat released (exothermic, temperature rises)
        """
        Q_dot = torch.zeros_like(list(rates.values())[0])
        
        # CO2 dissociation (endothermic - consumes heat)
        Q_dot -= rates['CO2_diss'] * self.reactions['CO2_diss']['delta_H']
        
        # N2 dissociation (very endothermic)
        Q_dot -= rates['N2_diss'] * self.reactions['N2_diss']['delta_H']
        
        # O2 dissociation (endothermic)
        Q_dot -= rates['O2_diss'] * self.reactions['O2_diss']['delta_H']
        
        # N2 + O reaction (endothermic)
        Q_dot -= rates['N2_O'] * self.reactions['N2_O']['delta_H']
        
        # O2 + N reaction (slightly exothermic)
        Q_dot -= rates['O2_N'] * self.reactions['O2_N']['delta_H']
        
        return Q_dot


class EnhancedChemicalPINN(nn.Module):
    """
    Enhanced PINN with reaction kinetics awareness
    """
    def __init__(self, input_dim=12, output_dim=11, hidden_layers=[124, 64, 24, 8]):
        super(EnhancedChemicalPINN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class EnhancedPhysicsLoss:
    """
    Enhanced physics loss with reaction kinetics
    """
    def __init__(self, species_names, molar_masses, atomic_composition):
        self.species_names = species_names
        self.n_species = len(species_names)
        self.molar_masses = molar_masses
        self.atomic_composition = atomic_composition
        self.R = 8.314
        
        # Create reaction kinetics calculator
        self.kinetics = ReactionKinetics()
    
    def atom_conservation_loss(self, X_input, X_output, rho_input, rho_output, T_input, T_output):
        """Atom conservation (same as before)"""
        batch_size = X_input.shape[0]
        
        W_mix_input = torch.zeros(batch_size, device=X_input.device)
        W_mix_output = torch.zeros(batch_size, device=X_output.device)
        
        for i, species in enumerate(self.species_names):
            W_i = self.molar_masses[species]
            W_mix_input += X_input[:, i] * W_i
            W_mix_output += X_output[:, i] * W_i
        
        n_total_input = rho_input / W_mix_input
        n_total_output = rho_output / W_mix_output
        
        atom_names = ['C', 'O', 'N', 'AR']
        losses = []
        
        for atom in atom_names:
            atom_count_input = torch.zeros(batch_size, device=X_input.device)
            atom_count_output = torch.zeros(batch_size, device=X_output.device)
            
            for i, species in enumerate(self.species_names):
                n_atoms = self.atomic_composition.get(species, {}).get(atom, 0)
                
                n_species_input = X_input[:, i] * n_total_input
                n_species_output = X_output[:, i] * n_total_output
                
                atom_count_input += n_atoms * n_species_input
                atom_count_output += n_atoms * n_species_output
            
            loss = torch.mean((atom_count_input - atom_count_output)**2)
            losses.append(loss)
        
        return sum(losses) / len(losses)
    
    def temperature_evolution_loss(self, X_input, X_output, T_input, T_output, rho_input, dt):
        """
        Temperature evolution based on reaction kinetics
        
        The temperature should change according to:
        ρ * Cp * dT/dt = -Σ(Q_reaction)
        
        where Q_reaction is the heat absorbed/released by reactions
        """
        # Compute reaction rates at input state
        rates, C_total = self.kinetics.compute_reaction_rates(X_input, T_input, rho_input)
        
        # Compute heat release rate (J/m³/s)
        Q_dot = self.kinetics.compute_heat_release(rates)
        
        # Specific heat capacity (approximate, J/(kg·K))
        # For high temperature air, Cp ≈ 1200-1400 J/(kg·K)
        Cp = 1300.0
        
        # Expected temperature change
        # dT = (Q_dot / (rho * Cp)) * dt
        dT_expected = (Q_dot / (rho_input.squeeze() * Cp)) * dt.squeeze()
        
        # Actual temperature change
        dT_actual = T_output.squeeze() - T_input.squeeze()
        
        # Loss: penalize deviation from physics-based temperature change
        loss = torch.mean((dT_actual - dT_expected)**2)
        
        return loss
    
    def dissociation_extent_loss(self, X_input, X_output, T_input, dt):
        """
        Enforce that dissociation extent matches reaction kinetics
        
        For CO2 dissociation: d[CO2]/dt = -k * [CO2] * [M]
        """
        batch_size = X_input.shape[0]
        
        # Compute reaction rates
        rates, C_total = self.kinetics.compute_reaction_rates(
            X_input, T_input, 
            torch.ones(batch_size, 1, device=X_input.device) * 0.0013  # Approximate rho
        )
        
        # Expected change in CO2 molar fraction
        # Rate is in mol/m³/s, convert to change in mole fraction
        dX_CO2_expected = -(rates['CO2_diss'] / C_total) * dt.squeeze()
        
        # Actual change
        dX_CO2_actual = X_output[:, 0] - X_input[:, 0]  # CO2 is first species
        
        # Loss
        loss = torch.mean((dX_CO2_actual - dX_CO2_expected)**2)
        
        return loss
    
    def volume_conservation_loss(self, rho_input, rho_output):
        """Constant volume: rho should be constant"""
        return torch.mean((rho_input - rho_output)**2)
    
    def sum_of_fractions_loss(self, X_output):
        """Molar fractions sum to 1"""
        sum_X = torch.sum(X_output[:, :self.n_species], dim=1)
        return torch.mean((sum_X - 1.0)**2)
    
    def non_negativity_loss(self, X_output):
        """Molar fractions must be non-negative"""
        return torch.mean(torch.relu(-X_output[:, :self.n_species])**2)


class EnhancedChemicalReactionPINN:
    """
    Enhanced training framework with reaction kinetics
    """
    def __init__(self, species_names, molar_masses, atomic_composition,
                 hidden_layers=[124, 64, 24, 8], device='cpu'):
        
        self.device = device
        self.species_names = species_names
        self.n_species = len(species_names)
        
        input_dim = self.n_species + 3
        output_dim = self.n_species + 2
        
        self.model = EnhancedChemicalPINN(input_dim, output_dim, hidden_layers).to(device)
        self.physics_loss = EnhancedPhysicsLoss(species_names, molar_masses, atomic_composition)
        
        # Loss weights
        self.lambda_data = 1.0
        self.lambda_atom = 1.0
        self.lambda_temp_evo = 0.5      # Temperature evolution from kinetics
        self.lambda_diss = 0.3           # Dissociation extent
        self.lambda_volume = 1.0
        self.lambda_sum = 10.0
        self.lambda_nonneg = 5.0
        
        self.optimizer = None
        self.scheduler = None
        
        self.loss_history = {
            'total': [], 'data': [], 'physics': [],
            'atom': [], 'temp_evo': [], 'diss': [],
            'volume': [], 'sum': [], 'nonneg': []
        }
    
    def data_loss(self, y_pred, y_true):
        return torch.mean((y_pred - y_true)**2)
    
    def compute_loss(self, inputs, outputs_true):
        """Compute total loss with enhanced physics"""
        outputs_pred = self.model(inputs)
        
        # Extract components
        X_input = inputs[:, :self.n_species]
        rho_input = inputs[:, self.n_species:self.n_species+1]
        T_input = inputs[:, self.n_species+1:self.n_species+2]
        dt = inputs[:, self.n_species+2:self.n_species+3]
        
        X_pred = outputs_pred[:, :self.n_species]
        rho_pred = outputs_pred[:, self.n_species:self.n_species+1]
        T_pred = outputs_pred[:, self.n_species+1:self.n_species+2]
        
        X_true = outputs_true[:, :self.n_species]
        rho_true = outputs_true[:, self.n_species:self.n_species+1]
        T_true = outputs_true[:, self.n_species+1:self.n_species+2]
        
        # Data loss
        loss_data = self.data_loss(outputs_pred, outputs_true)
        
        # Physics losses
        loss_atom = self.physics_loss.atom_conservation_loss(
            X_input, X_pred, rho_input, rho_pred, T_input, T_pred
        )
        
        loss_temp_evo = self.physics_loss.temperature_evolution_loss(
            X_input, X_pred, T_input, T_pred, rho_input, dt
        )
        
        loss_diss = self.physics_loss.dissociation_extent_loss(
            X_input, X_pred, T_input, dt
        )
        
        loss_volume = self.physics_loss.volume_conservation_loss(rho_input, rho_pred)
        loss_sum = self.physics_loss.sum_of_fractions_loss(X_pred)
        loss_nonneg = self.physics_loss.non_negativity_loss(X_pred)
        
        # Total physics loss
        loss_physics = (
            self.lambda_atom * loss_atom +
            self.lambda_temp_evo * loss_temp_evo +
            self.lambda_diss * loss_diss +
            self.lambda_volume * loss_volume +
            self.lambda_sum * loss_sum +
            self.lambda_nonneg * loss_nonneg
        )
        
        loss_total = self.lambda_data * loss_data + loss_physics
        
        losses = {
            'total': loss_total.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'atom': loss_atom.item(),
            'temp_evo': loss_temp_evo.item(),
            'diss': loss_diss.item(),
            'volume': loss_volume.item(),
            'sum': loss_sum.item(),
            'nonneg': loss_nonneg.item()
        }
        
        return loss_total, losses
    
    def train(self, train_loader, epochs=1000, learning_rate=0.001, 
              print_every=200, val_loader=None):
        """Train the enhanced PINN"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=300, verbose=True
        )
        
        print("Starting Enhanced PINN Training with Reaction Kinetics...")
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
                
                self.optimizer.zero_grad()
                loss, losses = self.compute_loss(inputs, outputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                for key, value in losses.items():
                    epoch_losses[key] += value
                n_batches += 1
            
            # Average losses
            for key in epoch_losses.keys():
                epoch_losses[key] /= n_batches
                self.loss_history[key].append(epoch_losses[key])
            
            self.scheduler.step(epoch_losses['total'])
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.1f}s")
                print(f"  Total Loss:   {epoch_losses['total']:.6e}")
                print(f"  Data Loss:    {epoch_losses['data']:.6e}")
                print(f"  Physics Loss: {epoch_losses['physics']:.6e}")
                print(f"    - Atom:     {epoch_losses['atom']:.6e}")
                print(f"    - Temp Evo: {epoch_losses['temp_evo']:.6e}")
                print(f"    - Diss:     {epoch_losses['diss']:.6e}")
                print(f"    - Volume:   {epoch_losses['volume']:.6e}")
                print("-" * 80)
        
        print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    
    def predict(self, inputs):
        """Make predictions"""
        self.model.eval()
        
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return outputs.cpu().numpy()
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total and data loss
        ax = axes[0, 0]
        ax.semilogy(self.loss_history['total'], label='Total', linewidth=2)
        ax.semilogy(self.loss_history['data'], label='Data', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total and Data Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Physics components
        ax = axes[0, 1]
        ax.semilogy(self.loss_history['atom'], label='Atom', linewidth=2)
        ax.semilogy(self.loss_history['temp_evo'], label='Temp Evolution', linewidth=2)
        ax.semilogy(self.loss_history['diss'], label='Dissociation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Reaction Kinetics Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Other constraints
        ax = axes[1, 0]
        ax.semilogy(self.loss_history['volume'], label='Volume', linewidth=2)
        ax.semilogy(self.loss_history['sum'], label='Sum=1', linewidth=2)
        ax.semilogy(self.loss_history['nonneg'], label='Non-neg', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Conservation Constraints')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # All physics
        ax = axes[1, 1]
        ax.semilogy(self.loss_history['physics'], label='Total Physics', 
                   linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Physics Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Test with dummy data
if __name__ == "__main__":
    print("Enhanced Chemical PINN with Reaction Kinetics")
    print("=" * 80)
    print("\nThis module includes:")
    print("✓ Arrhenius rate calculations for major reactions")
    print("✓ Temperature evolution from heat release")
    print("✓ Dissociation extent matching kinetics")
    print("✓ All conservation laws")
    print("\nReady to train with your Cantera data!")
