"""
Demo script with synthetic chemical reaction data
This creates realistic-looking data for demonstration purposes
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_reaction_data(n_points=1000, save_path='reaction_data.npz'):
    """
    Generate synthetic chemical reaction data that mimics
    a realistic combustion reaction (CO2 dissociation)
    
    Simplified reaction: CO2 → CO + 0.5 O2 (high temperature dissociation)
    """
    print("Generating synthetic chemical reaction data...")
    print(f"Number of points: {n_points}")
    
    # Time array (logarithmic spacing for reaction kinetics)
    t_start = 1e-8
    t_end = 1e-2
    time = np.logspace(np.log10(t_start), np.log10(t_end), n_points)
    
    # Initial conditions
    T_initial = 7500  # K (high temperature)
    rho_initial = 0.0013  # kg/m³
    
    # Initial species molar fractions (Mars-like atmosphere)
    X_CO2_init = 0.9556
    X_N2_init = 0.0270
    X_AR_init = 0.0160
    X_O2_init = 0.0014
    
    # Reaction rate parameters (arbitrary for demonstration)
    k = 50.0  # reaction rate constant
    T_eq = 4000  # Equilibrium temperature
    
    # Initialize arrays
    X_CO2 = np.zeros(n_points)
    X_CO = np.zeros(n_points)
    X_O2 = np.zeros(n_points)
    X_N2 = np.zeros(n_points)
    X_AR = np.zeros(n_points)
    X_C = np.zeros(n_points)
    X_O = np.zeros(n_points)
    X_N = np.zeros(n_points)
    X_NO = np.zeros(n_points)
    
    Temperature = np.zeros(n_points)
    Density = np.zeros(n_points)
    
    # Set initial conditions
    X_CO2[0] = X_CO2_init
    X_N2[0] = X_N2_init
    X_AR[0] = X_AR_init
    X_O2[0] = X_O2_init
    Temperature[0] = T_initial
    Density[0] = rho_initial
    
    # Evolve system
    for i in range(1, n_points):
        dt = time[i] - time[i-1]
        
        # Simple dissociation kinetics
        # CO2 → CO + 0.5 O2
        extent = k * dt * np.exp(-5000/Temperature[i-1]) * X_CO2[i-1]
        extent = min(extent, X_CO2[i-1] * 0.1)  # Limit reaction rate
        
        # Update species
        X_CO2[i] = X_CO2[i-1] - extent
        X_CO[i] = X_CO[i-1] + extent
        X_O2[i] = X_O2[i-1] + 0.5 * extent
        
        # Inert species remain constant
        X_N2[i] = X_N2[i-1]
        X_AR[i] = X_AR[i-1]
        
        # Small amounts of radicals (for realism)
        X_C[i] = extent * 0.001 * np.exp(-8000/Temperature[i-1])
        X_O[i] = extent * 0.002 * np.exp(-8000/Temperature[i-1])
        X_N[i] = X_N2[i] * 0.0001 * np.exp(-10000/Temperature[i-1])
        X_NO[i] = X_N2[i] * X_O2[i] * 0.01 * np.exp(-5000/Temperature[i-1])
        
        # Normalize to ensure sum = 1
        total = X_CO2[i] + X_CO[i] + X_O2[i] + X_N2[i] + X_AR[i] + \
                X_C[i] + X_O[i] + X_N[i] + X_NO[i]
        
        X_CO2[i] /= total
        X_CO[i] /= total
        X_O2[i] /= total
        X_N2[i] /= total
        X_AR[i] /= total
        X_C[i] /= total
        X_O[i] /= total
        X_N[i] /= total
        X_NO[i] /= total
        
        # Temperature evolution (cooling towards equilibrium)
        dT = -(Temperature[i-1] - T_eq) * 0.1 * dt / (time[i] + 1e-10)
        Temperature[i] = Temperature[i-1] + dT
        
        # Density (approximately constant for constant volume)
        # Small variations due to temperature changes
        Density[i] = rho_initial * T_initial / Temperature[i]
    
    # Stack species
    X_history = np.column_stack([
        X_CO2, X_O2, X_N2, X_CO, X_NO, X_C, X_O, X_N, X_AR
    ])
    
    # Create PINN training data
    # Input: [X(t), rho(t), T(t), dt]
    # Output: [X(t+dt), rho(t+dt), T(t+dt)]
    
    inputs_list = []
    outputs_list = []
    dt_list = []
    
    for i in range(n_points - 1):
        # Input state
        state_in = np.concatenate([
            X_history[i],      # 9 species
            [Density[i]],      # rho
            [Temperature[i]]   # T
        ])
        
        # Output state
        state_out = np.concatenate([
            X_history[i+1],
            [Density[i+1]],
            [Temperature[i+1]]
        ])
        
        # Time step
        dt_current = time[i+1] - time[i]
        
        # Full input with dt
        input_full = np.concatenate([state_in, [dt_current]])
        
        inputs_list.append(input_full)
        outputs_list.append(state_out)
        dt_list.append(dt_current)
    
    # Convert to arrays
    inputs_array = np.array(inputs_list, dtype=np.float32)
    outputs_array = np.array(outputs_list, dtype=np.float32)
    dt_array = np.array(dt_list, dtype=np.float32).reshape(-1, 1)
    
    print(f"\nData shapes:")
    print(f"  Inputs: {inputs_array.shape}")
    print(f"  Outputs: {outputs_array.shape}")
    
    # Save data
    species_names = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']
    
    np.savez(save_path,
             inputs=inputs_array,
             outputs=outputs_array,
             dt=dt_array,
             species_names=np.array(species_names),
             time=time,
             T_equilibrium=T_eq)
    
    print(f"\nData saved to: {save_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Species evolution
    ax = axes[0]
    colors = ['r', 'g', 'k', 'm', 'c', 'orange', 'purple', 'pink', 'brown']
    
    for i, name in enumerate(species_names):
        log_X = np.log10(np.maximum(X_history[:, i], 1e-16))
        ax.semilogx(time, log_X, color=colors[i], linewidth=2, label=name)
    
    ax.set_xlabel('Time (s) - Log Scale', fontsize=12)
    ax.set_ylabel('log₁₀(Molar Fraction)', fontsize=12)
    ax.set_title('Synthetic Species Evolution', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Temperature
    ax = axes[1]
    ax.semilogx(time, Temperature, 'r-', linewidth=3, label='Temperature')
    ax.axhline(T_eq, color='blue', linestyle='--', label=f'Equilibrium ({T_eq} K)')
    ax.set_xlabel('Time (s) - Log Scale', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('Temperature Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return inputs_array, outputs_array, time, X_history, Temperature, Density


if __name__ == "__main__":
    print("="*80)
    print("Synthetic Chemical Reaction Data Generator")
    print("="*80)
    
    inputs, outputs, time, X, T, rho = generate_synthetic_reaction_data(
        n_points=5000,
        save_path='reaction_data.npz'
    )
    
    print("\n" + "="*80)
    print("Synthetic data generation complete!")
    print("You can now train the PINN with: python chemical_pinn.py")
    print("="*80)
