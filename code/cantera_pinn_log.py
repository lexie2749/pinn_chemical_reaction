#%% Import necessary libraries
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

#%% Set physical model case
####################################################################################################################

# Set initial gas conditions
T_0 = 7500 # K - Initial temperature reaction
rho_0 = 0.0013 # kg/m³ - Initial density reaction

E_const = 0 # J/kg - Energy offset to match reference data if needed

# Set planet atmosphere
q = 'CO2:0.9556,N2:0.0270,Ar:0.0160,O2:0.0014' # Composition, mole fractions
mech = '/Users/xiaoxizhou/Downloads/su_26/adrian_surf/airNASA9ions.yaml' # NASA 9 no ions species mechanism

# Select fixed variables for equilibrium calculations
fixed_vars = 'UV'  # Internal Energy-Volume

print(f"Main mechanism: {mech}")

#%% Single case analysis
####################################################################################################################
print(f"\nSingle point analysis:")

# Create a reactor for time evolution
gas_react = ct.Solution(mech)

# Obtain next of initial conditions from initial gas state
gas_react.X = q  # Set initial mole fractions
gas_react.TD = T_0, rho_0
v_0 = gas_react.v  # m³/kg
P_0 = gas_react.P  # Pa
e_0 = gas_react.u + E_const  
h_0 = gas_react.h + E_const 
print(f"T_0 = {T_0} K, rho_0 = {rho_0} kg/m³", f"=> P_0 = {P_0/1000:.1f} kPa")

# Choose evolution type based on fixed_vars
if fixed_vars == 'HP':
    reactor = ct.ConstPressureReactor(gas_react)
    print(f"Reactor type: Constant Pressure Reactor (HP)")
elif fixed_vars == 'UV':
    reactor = ct.IdealGasReactor(gas_react)
    reactor.volume = v_0  
    print(f"Reactor type: Constant Volume Reactor (UV)")

reactor_net = ct.ReactorNet([reactor])

# Calculate equilibrium reference state for comparison
gas_equilibrium = ct.Solution(mech)
gas_equilibrium.X = q
if fixed_vars == 'HP':
    gas_equilibrium.HP = h_0 - E_const, P_0  
    gas_equilibrium.equilibrate('HP')
elif fixed_vars == 'UV':
    gas_equilibrium.UV = e_0, v_0
    gas_equilibrium.equilibrate('UV')

T_equilibrium = gas_equilibrium.T 

print(f"Equilibrium temperature: {T_equilibrium:.1f} K")

# Time integration parameters
t_end = 1e-2 * np.exp(2000/T_0) * (0.1/rho_0**1.5) 
dt = 1e-14 * np.exp(2000/T_0) * (0.1/rho_0**1.5) 
n_points= int(20000) 
time = np.logspace(np.log10(dt), np.log10(t_end), n_points)

# ==========================================
# 1. Initialize data containers
# ==========================================

# Plotting arrays
TEMP = np.zeros(n_points)
PRESSURE = np.zeros(n_points)
DENSITY = np.zeros(n_points)
ENERGY = np.zeros(n_points)
ENTHALPY = np.zeros(n_points)

# Species names and indices
species_names = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']
species_indices = {}
pinn_indices = []

print("Identifying Species Indices...")
for name in species_names:
    try:
        idx = gas_react.species_index(name)
        species_indices[name] = idx
        pinn_indices.append(idx)
    except ValueError:
        print(f"Warning: Species {name} not found in mechanism!")
        species_indices[name] = None
        pinn_indices.append(0)

# Plotting arrays (Log10)
log_data = {name: np.zeros(n_points) for name in species_names}

# PINN training data (Raw history arrays)
# Shape: (TimeSteps, Num_Species)
X_pinn_history = np.zeros((n_points, len(species_names))) 
rho_history = np.zeros(n_points)
T_history = np.zeros(n_points)
time_history = np.zeros(n_points)

print("Starting time evolution...")
print(f"Initial conditions: T = {T_0} K, P = {P_0/1000:.1f} kPa")

# ==========================================
# 2. Time evolution loop
# ==========================================
for i in range(n_points):
    # Advance time
    reactor_net.advance(time[i])

    # Get current state
    current_temp = reactor.thermo.T
    current_rho = reactor.thermo.density
    current_X = reactor.thermo.X
    
    # Store plotting data
    TEMP[i] = current_temp
    PRESSURE[i] = reactor.thermo.P
    DENSITY[i] = current_rho
    ENERGY[i] = reactor.thermo.u + E_const
    ENTHALPY[i] = reactor.thermo.h + E_const
    
    # Calculate Log10 mole fractions (for plotting)
    min_fraction = 1e-16
    for name in species_names:
        idx = species_indices[name]
        if idx is not None:
            log_data[name][i] = np.log10(max(current_X[idx], min_fraction))
        else:
            log_data[name][i] = -16

    # Store PINN raw data
    for k, idx in enumerate(pinn_indices):
        X_pinn_history[i, k] = current_X[idx]
    
    rho_history[i] = current_rho
    T_history[i] = current_temp
    time_history[i] = time[i]

print("Time evolution completed!")

# ==========================================
# 3. Plots
# ==========================================
plt.figure(figsize=(14, 8))
colors = ['r', 'g', 'k', 'm', 'c', 'orange', 'purple', 'pink', 'brown']
for k, name in enumerate(species_names):
    plt.semilogx(time, log_data[name], color=colors[k % len(colors)], linewidth=2, label=name)

plt.xlabel('Time (s) - Log Scale')
plt.ylabel('log₁₀(Molar Fraction)')
plt.title(f'Species Evolution')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.semilogx(time, TEMP, 'r-', linewidth=3, label='Temperature Evolution')
plt.axhline(T_equilibrium, color='blue', linestyle='-', alpha=0.7, 
           label=f'Equilibrium ({T_equilibrium:.1f} K)')
plt.xlabel('Time (s) - Log Scale')
plt.ylabel('Temperature (K)')
plt.title('Temperature Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 4. Process and save PINN data (.npz) WITH LOG SCALING
# ==========================================
print("\nProcessing data for PINN with logarithmic inputs...")

inputs_list = []
outputs_list = []

# Minimum values for log transformation (to avoid log(0))
MIN_CONC = 1e-20
MIN_RHO = 1e-6
MIN_T = 1.0
MIN_DT = 1e-20

# Build (t) -> (t+dt) data pairs
# We only have n_points points, so we can only build n_points-1 pairs
for i in range(n_points - 1):
    # ===== INPUT STATE (at time i) =====
    # Convert concentrations to log10 scale
    log_X_in = np.log10(np.maximum(X_pinn_history[i], MIN_CONC))
    log_rho_in = np.log10(max(rho_history[i], MIN_RHO))
    log_T_in = np.log10(max(T_history[i], MIN_T))
    
    # Time step in log10 scale
    dt_current = time_history[i+1] - time_history[i]
    if dt_current <= 0:
        continue
    log_dt = np.log10(max(dt_current, MIN_DT))
    
    # Input vector: [log10(X1), ..., log10(X9), log10(rho), log10(T), log10(dt)]
    state_in = np.concatenate([log_X_in, [log_rho_in], [log_T_in], [log_dt]])
    
    # ===== OUTPUT STATE (at time i+1) =====
    # Also in log10 scale for consistency
    log_X_out = np.log10(np.maximum(X_pinn_history[i+1], MIN_CONC))
    log_rho_out = np.log10(max(rho_history[i+1], MIN_RHO))
    log_T_out = np.log10(max(T_history[i+1], MIN_T))
    
    # Output vector: [log10(X1), ..., log10(X9), log10(rho), log10(T)]
    state_out = np.concatenate([log_X_out, [log_rho_out], [log_T_out]])
    
    inputs_list.append(state_in)
    outputs_list.append(state_out)

# Convert to arrays
inputs_array = np.array(inputs_list, dtype=np.float32)
outputs_array = np.array(outputs_list, dtype=np.float32)

# Save
save_filename = 'reaction_data_log.npz'
np.savez(save_filename, 
         inputs=inputs_array, 
         outputs=outputs_array,
         species_names=np.array(species_names),
         initial_conditions={'T_0': T_0, 'rho_0': rho_0, 'composition': q})

print(f"\nData saved to {save_filename}")
print(f"Input shape: {inputs_array.shape}")  # Expected: (~19999, 12)
print(f"  - 9 species concentrations (log10)")
print(f"  - 1 density (log10)")
print(f"  - 1 temperature (log10)")
print(f"  - 1 time step (log10)")
print(f"Output shape: {outputs_array.shape}") # Expected: (~19999, 11)
print(f"  - 9 species concentrations (log10)")
print(f"  - 1 density (log10)")
print(f"  - 1 temperature (log10)")

# Print example data statistics
print("\n" + "="*60)
print("Data Statistics (in log10 scale)")
print("="*60)
print("\nInput ranges:")
for i, name in enumerate(species_names):
    print(f"  log10({name:4s}): [{inputs_array[:, i].min():.2f}, {inputs_array[:, i].max():.2f}]")
print(f"  log10(rho):  [{inputs_array[:, 9].min():.2f}, {inputs_array[:, 9].max():.2f}]")
print(f"  log10(T):    [{inputs_array[:, 10].min():.2f}, {inputs_array[:, 10].max():.2f}]")
print(f"  log10(dt):   [{inputs_array[:, 11].min():.2f}, {inputs_array[:, 11].max():.2f}]")

print("\nOutput ranges:")
for i, name in enumerate(species_names):
    print(f"  log10({name:4s}): [{outputs_array[:, i].min():.2f}, {outputs_array[:, i].max():.2f}]")
print(f"  log10(rho):  [{outputs_array[:, 9].min():.2f}, {outputs_array[:, 9].max():.2f}]")
print(f"  log10(T):    [{outputs_array[:, 10].min():.2f}, {outputs_array[:, 10].max():.2f}]")

print("\n" + "="*60)
print("Key advantages of log-scale inputs:")
print("="*60)
print("1. Time steps: log10(1e-14) = -14.0 vs log10(1e-2) = -2.0")
print("   → Clear distinction between fast/slow reactions")
print("2. Concentrations: log10(1e-16) = -16.0 vs log10(1.0) = 0.0")
print("   → Captures trace species dynamics")
print("3. Neural network sees meaningful differences at all scales")
print("   → Better gradient flow and convergence")
