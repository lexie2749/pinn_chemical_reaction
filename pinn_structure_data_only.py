import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 设备与网络结构
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class ReactionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 训练模型 (纯数据驱动)
# ==========================================
def train_model():
    # 归一化系数 (必须与数据特征匹配)
    SCALE_T = 8000.0   
    SCALE_RHO = 0.002  
    SCALE_DT = 1e-5    
    # 显式定义初始物理量（Cantera定义的值）
    initial_X = np.array([0.9556, 0.0014, 0.0270, 0, 0, 0, 0, 0, 0.0160]) # CO2, O2, N2, CO, NO, C, O, N, AR
    initial_T = 7500.0
    initial_rho = 0.0013

    # 即使在循环开始，也要确保初始点 (t=0) 是准确的
    # 不要让模型预测第0步，直接硬编码第0步的数据
    X_history[0] = initial_X
    T_history[0] = initial_T
    
    try:
        data = np.load('reaction_data.npz')
        inputs_np = data['inputs']   
        outputs_np = data['outputs'] 
        
        inputs = torch.tensor(inputs_np, dtype=torch.float32).to(device)
        targets = torch.tensor(outputs_np, dtype=torch.float32).to(device)

        # 归一化输入: [0:9]=X, [9]=rho, [10]=T, [11]=dt
        inputs[:, 9]  /= SCALE_RHO
        inputs[:, 10] /= SCALE_T
        inputs[:, 11] /= SCALE_DT 
        # 归一化输出标签: [0:9]=X, [9]=rho, [10]=T
        targets[:, 9]  /= SCALE_RHO
        targets[:, 10] /= SCALE_T
        
    except FileNotFoundError:
        print("Error: reaction_data.npz 未找到，请先运行 Cantera 脚本生成数据。")
        return None

    model = ReactionNet(12, 11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    print("开始训练神经网络 (对比 Cantera 数据)...")
    for epoch in range(3001): # 3000代通常足够看到初步结果
        model.train()
        optimizer.zero_grad()
        
        pred = model(inputs)
        
        # 物种部分加 Softmax 保证物理合理性
        X_pred = torch.nn.functional.softmax(pred[:, :9], dim=1)
        rho_T_pred = pred[:, 9:11]
        
        loss = criterion(X_pred, targets[:, :9]) + criterion(rho_T_pred, targets[:, 9:11])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.8f}")

    return model

# ==========================================
# 3. 绘图对比: Cantera vs ML
# ==========================================
def plot_comparison(model, data_path='reaction_data.npz'):
    if model is None: return

    # 1. 加载 Cantera 原始数据 (Truth)
    data = np.load(data_path)
    # inputs 存储的是 [X_i, rho_i, T_i, dt_i]
    # outputs 存储的是 [X_{i+1}, rho_{i+1}, T_{i+1}]
    cantera_inputs = data['inputs']
    cantera_outputs = data['outputs']
    dt_array = data['dt'].flatten()
    
    # 提取 Cantera 的真实演化序列
    # 时间轴：从 dt[0] 开始累加
    time_truth = np.cumsum(dt_array)
    # 真实值：outputs 每一行就是下一步的真实状态
    X_truth = cantera_outputs[:, :9]
    T_truth = cantera_outputs[:, 10]

    # 2. ML 自回归预测 (Prediction)
    SCALE_T = 8000.0
    SCALE_RHO = 0.002
    SCALE_DT = 1e-5
    species_names = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']

    n_steps = len(cantera_inputs)
    X_pred_hist = np.zeros((n_steps, 9))
    T_pred_hist = np.zeros(n_steps)

    # 初始状态设为 Cantera 数据的第一行输入
    current_state_phys = cantera_inputs[0, :11].copy()
    
    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            # 归一化输入
            state_norm = current_state_phys.copy()
            state_norm[9]  /= SCALE_RHO
            state_norm[10] /= SCALE_T
            dt_n = dt_array[i] / SCALE_DT
            
            net_input = torch.tensor(np.append(state_norm, dt_n), dtype=torch.float32).unsqueeze(0).to(device)
            
            # 模型预测
            pred = model(net_input)
            
            # 反归一化并保存
            X_next = torch.nn.functional.softmax(pred[:, :9], dim=1).cpu().numpy()[0]
            rho_next = pred[0, 9].item() * SCALE_RHO
            T_next = pred[0, 10].item() * SCALE_T
            
            X_pred_hist[i] = X_next
            T_pred_hist[i] = T_next
            
            # 更新状态用于下一步
            current_state_phys = np.concatenate([X_next, [rho_next], [T_next]])

    # 3. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, 9))

    # --- 物种对比图 ---
    for k, name in enumerate(species_names):
        # Cantera 真实值用点或虚线
        ax1.semilogx(time_truth, np.log10(X_truth[:, k] + 1e-15), '--', color=colors[k], alpha=0.6)
        # ML 预测值用实线
        ax1.semilogx(time_truth, np.log10(X_pred_hist[:, k] + 1e-15), '-', color=colors[k], label=name)
    
    ax1.set_title('Species Evolution: Cantera (Dashed) vs ML (Solid)')
    ax1.set_ylabel('log10(Molar Fraction)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, which='both', alpha=0.2)

    # --- 温度对比图 ---
    ax2.semilogx(time_truth, T_truth, 'k--', linewidth=2, label='Cantera Truth', alpha=0.7)
    ax2.semilogx(time_truth, T_pred_hist, 'r-', linewidth=2, label='ML Prediction')
    ax2.set_title('Temperature Evolution: Cantera vs ML')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Temperature (K)')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 执行流程
    model = train_model()
    plot_comparison(model)