import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 配置与定义
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

class ReactionPINN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def generate_dummy_data():
    """如果不提供数据文件，生成一些符合物理维度的随机假数据用于演示"""
    print("Warning: Generating dummy data for demonstration...")
    N_samples = 1000
    # Inputs: [Species(9), rho, T, dt] -> 12 dims
    # Outputs: [Species(9), rho, T] -> 11 dims
    
    # 模拟简单的衰减/增长曲线
    t = np.linspace(0, 0.01, N_samples)
    dt = np.diff(t, prepend=0)
    dt[0] = dt[1]
    
    # 构造假的状态数据
    species = np.abs(np.sin(t.reshape(-1, 1) * np.arange(1, 10))) 
    species = species / species.sum(axis=1, keepdims=True) # 归一化
    
    rho = 1.0 - 0.1 * t
    T = 2000.0 + 500.0 * t
    
    inputs = np.column_stack([species[:-1], rho[:-1], T[:-1], dt[:-1]])
    outputs = np.column_stack([species[1:], rho[1:], T[1:]])
    
    return inputs, outputs, dt[:-1]

# ==========================================
# 1. 训练与预测主程序
# ==========================================
def run_pinn_analysis():
    # --- 归一化参数 ---
    SCALE_T = 8000.0   
    SCALE_RHO = 0.002  
    SCALE_DT = 1e-5    
    
    # 维度
    inputs_dim = 12 
    outputs_dim = 11
    
    model = ReactionPINN(inputs_dim, outputs_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # --- 加载数据 ---
    print("Loading data...")
    if os.path.exists('reaction_data.npz'):
        data = np.load('reaction_data.npz')
        inputs_raw = data['inputs']   
        outputs_raw = data['outputs'] 
        dt_array_raw = data['dt']
    else:
        inputs_raw, outputs_raw, dt_array_raw = generate_dummy_data()
        
    # 转换为 Tensor 用于训练
    inputs_train = torch.tensor(inputs_raw, dtype=torch.float32).to(device)
    targets_train = torch.tensor(outputs_raw, dtype=torch.float32).to(device)
    
    # === 训练前原地归一化 ===
    # Inputs: [X(9), rho, T, dt]
    # 注意：这里使用 clone() 避免修改原始 raw 数据，因为后面画图要用 raw 数据
    inputs_train_norm = inputs_train.clone()
    targets_train_norm = targets_train.clone()

    inputs_train_norm[:, 9]  /= SCALE_RHO
    inputs_train_norm[:, 10] /= SCALE_T
    inputs_train_norm[:, 11] /= SCALE_DT 

    # Targets: [X(9), rho, T]
    targets_train_norm[:, 9]  /= SCALE_RHO
    targets_train_norm[:, 10] /= SCALE_T

    # --- 训练循环 ---
    print(f"Starting Training...")
    epochs = 2000 
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        pred = model(inputs_train_norm)
        
        # 物理约束：Softmax 确保组分和为1
        X_pred_logits = pred[:, :9]
        X_pred = torch.nn.functional.softmax(X_pred_logits, dim=1)
        
        pred_final = torch.cat([X_pred, pred[:, 9:]], dim=1)
        
        loss = loss_fn(pred_final, targets_train_norm)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    print("Training Completed.")

    # ==========================================
    # 2. Time Integration (Rollout) - 核心部分
    # ==========================================
    print("\nStarting Time Integration (Rollout Prediction)...")
    
    # 获取初始条件 (t=0)
    current_state_phys = inputs_raw[0, :11].copy() 
    pred_history = [current_state_phys.copy()]
    
    # 获取 dt 序列
    dt_sequence = dt_array_raw.flatten() if isinstance(dt_array_raw, np.ndarray) else dt_array_raw
    
    model.eval() 
    
    # 【修复重点】：以下代码块现在正确缩进在 run_pinn_analysis 函数内
    with torch.no_grad():
        for i, dt_val in enumerate(dt_sequence):
            # 1. 准备输入向量
            inp = torch.zeros(1, 12).to(device)
            
            # 填入 X
            inp[0, :9] = torch.tensor(current_state_phys[:9], dtype=torch.float32).to(device)
            
            # 填入 rho, T (归一化)
            inp[0, 9]  = float(current_state_phys[9] / SCALE_RHO)
            inp[0, 10] = float(current_state_phys[10] / SCALE_T)
            
            # 填入 dt (归一化)
            inp[0, 11] = float(dt_val / SCALE_DT)
            
            # 2. PINN 预测
            out_norm = model(inp)
            
            # 3. 解析并反归一化
            pred_X = torch.nn.functional.softmax(out_norm[0, :9], dim=0).cpu().numpy()
            pred_rho = out_norm[0, 9].item() * SCALE_RHO
            pred_T   = out_norm[0, 10].item() * SCALE_T
            
            # 4. 更新状态
            next_state_phys = np.concatenate([pred_X, [pred_rho], [pred_T]])
            pred_history.append(next_state_phys)
            current_state_phys = next_state_phys

    # 转换为数组
    pred_history = np.array(pred_history)
    
    # 构建 Ground Truth (注意维度匹配，取前 N+1 个点)
    # 因为 inputs_raw[0] 是 t0, outputs_raw 是 t1...tn
    # 如果 dt_sequence 长度为 N，我们需要 N+1 个点
    limit = len(dt_sequence) + 1
    
    # 拼接初始状态和后续真实状态
    gt_history = np.vstack([inputs_raw[0, :11], outputs_raw[:limit-1, :11]])
    
    # 时间轴
    time_accumulated = np.concatenate(([0], np.cumsum(dt_sequence)))

    # ==========================================
    # 3. 对比绘图
    # ==========================================
    print("Generating Plots...")
    species_names = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']
    
    # --- 图 1: 温度对比 ---
    plt.figure(figsize=(10, 6))
    plt.semilogx(time_accumulated, gt_history[:, 10], 'k-', linewidth=3, alpha=0.6, label='Ground Truth')
    plt.semilogx(time_accumulated, pred_history[:, 10], 'r--', linewidth=2.5, label='PINN Prediction')
    plt.title('Temperature Evolution')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()

    # --- 图 2: 组分对比 ---
    # 根据实际存在的组分数量进行绘图
    num_species_to_plot = min(6, len(species_names))
    plot_species = species_names[:num_species_to_plot]
    colors = ['r', 'b', 'g', 'orange', 'purple', 'brown']
    
    plt.figure(figsize=(14, 8))
    for i, sp_name in enumerate(plot_species):
        color = colors[i % len(colors)]
        
        # Ground Truth
        y_gt = np.log10(np.maximum(gt_history[:, i], 1e-20))
        plt.semilogx(time_accumulated, y_gt, '-', color=color, linewidth=3, alpha=0.5, 
                    label=f'{sp_name}' if i==0 else "") # 只标一次图例占位
        
        # PINN
        y_pred = np.log10(np.maximum(pred_history[:, i], 1e-20))
        plt.semilogx(time_accumulated, y_pred, '--', color=color, linewidth=2, 
                    label=f'{sp_name}')

    plt.title('Species Molar Fraction Evolution (Log Scale)')
    plt.xlabel('Time (s)')
    plt.ylabel('log10(Mole Fraction)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # 图例
    from matplotlib.lines import Line2D
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.text(1.02, 0.5, "Solid: Ground Truth\nDashed: PINN", transform=plt.gca().transAxes, fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pinn_analysis()