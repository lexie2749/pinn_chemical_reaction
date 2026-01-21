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
mech = '/Users/xiaoxizhou/Downloads/surf/adrian_surf/airNASA9ions.yaml' # NASA 9 no ions species mechanism

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
# 1. 初始化数据容器 (合并了绘图数据和 PINN 数据)
# ==========================================

# 绘图用 (Plotting Arrays)
TEMP = np.zeros(n_points)
PRESSURE = np.zeros(n_points)
DENSITY = np.zeros(n_points)
ENERGY = np.zeros(n_points)
ENTHALPY = np.zeros(n_points)

# 物种名称与索引查找
species_names = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR'] # 添加了 AR
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
        pinn_indices.append(0) # 占位

# 绘图用 (Log10 Arrays)
log_data = {name: np.zeros(n_points) for name in species_names}

# PINN 训练用 (Raw History Arrays)
# 形状: (TimeSteps, Num_Species)
X_pinn_history = np.zeros((n_points, len(species_names))) 
rho_history = np.zeros(n_points)
T_history = np.zeros(n_points)
time_history = np.zeros(n_points)

print("Starting time evolution...")
print(f"Initial conditions: T = {T_0} K, P = {P_0/1000:.1f} kPa")

# ==========================================
# 2. 统一的时间演化循环 (核心修复)
# ==========================================
for i in range(n_points):
    # 1. 推进时间
    reactor_net.advance(time[i])

    # 2. 获取当前状态
    # 注意：Cantera 3.2+ 建议使用 reactor.thermo，但在旧版本直接属性访问也行
    current_temp = reactor.thermo.T
    current_rho = reactor.thermo.density
    current_X = reactor.thermo.X
    
    # 3. 存储绘图数据
    TEMP[i] = current_temp
    PRESSURE[i] = reactor.thermo.P
    DENSITY[i] = current_rho
    ENERGY[i] = reactor.thermo.u + E_const
    ENTHALPY[i] = reactor.thermo.h + E_const
    
    # 计算 Log10 摩尔分数 (用于绘图)
    min_fraction = 1e-16
    for name in species_names:
        idx = species_indices[name]
        if idx is not None:
            log_data[name][i] = np.log10(max(current_X[idx], min_fraction))
        else:
            log_data[name][i] = -16

    # 4. 存储 PINN 原始数据 (用于神经网络)
    # 按照 species_names 的顺序提取数据
    for k, idx in enumerate(pinn_indices):
        X_pinn_history[i, k] = current_X[idx]
    
    rho_history[i] = current_rho
    T_history[i] = current_temp
    time_history[i] = time[i]

print("Time evolution completed!")

# ==========================================
# 3. 绘图 (Plots)
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
# 4. 处理并保存 PINN 数据 (.npz)
# ==========================================
print("\nProcessing data for PINN...")

inputs_list = []
outputs_list = []
dt_list = []

# 构建 (t) -> (t+dt) 数据对
# 我们只有 n_points 个点，所以只能构建 n_points-1 对数据
for i in range(n_points - 1):
    # Input State: [X... , rho, T] at time i
    state_in = np.concatenate([X_pinn_history[i], [rho_history[i]], [T_history[i]]])
    
    # Output State: [X... , rho, T] at time i+1
    state_out = np.concatenate([X_pinn_history[i+1], [rho_history[i+1]], [T_history[i+1]]])
    
    # Dt: time[i+1] - time[i]
    dt_current = time_history[i+1] - time_history[i]
    
    # 如果 dt 为 0 或负数 (数值误差)，跳过
    if dt_current <= 0:
        continue
        
    # Input vector needed for PINN: [State_in, dt]
    inputs_list.append(np.concatenate([state_in, [dt_current]]))
    outputs_list.append(state_out)
    dt_list.append(dt_current)

# Convert to arrays
inputs_array = np.array(inputs_list, dtype=np.float32)
outputs_array = np.array(outputs_list, dtype=np.float32)
dt_array = np.array(dt_list, dtype=np.float32).reshape(-1, 1)

# Save
save_filename = 'reaction_data.npz'
np.savez(save_filename, inputs=inputs_array, outputs=outputs_array, dt=dt_array)

print(f"Data saved to {save_filename}")
print(f"Input shape: {inputs_array.shape}") # Expected: (~19999, 9_species + 1_rho + 1_T + 1_dt = 12)
print(f"Output shape: {outputs_array.shape}") # Expected: (~19999, 11)