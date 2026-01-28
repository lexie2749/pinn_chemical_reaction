import torch
from sequential_pinn_log import Config

# Register the class so PyTorch trusts it during unpickling
torch.serialization.add_safe_globals([Config])

# Now load the model
checkpoint = torch.load(
    '/Users/xiaoxizhou/Downloads/su_26/adrian_surf/outputs_sequential_log/best_model.pt', 
    map_location=torch.device('cpu'),
    weights_only=False 
)

print(checkpoint)