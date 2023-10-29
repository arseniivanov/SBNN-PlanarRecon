from data_loader import get_loaders
import torch
from snntorch import utils
import numpy as np
from snntorch import surrogate
from sklearn.metrics import accuracy_score
from architectures import RecurrentSNN_v2
from config import num_steps, batch_size, num_classes, beta, spike_grad, net

if spike_grad == "fast_sigmoid":
    spike_grad = surrogate.fast_sigmoid()

def evaluate_and_visualize(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            utils.reset(model)
            
            data = data.permute(1, 0, 2, 3, 4)  # Permute for time steps (timesteps, batch, C, W, H)

            spike = model(data)
            
            all_preds.append(spike.argmax(dim=1).cpu().numpy())
            all_labels.append(target.cpu().numpy())
                
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    test_accuracy = accuracy_score(all_preds, all_labels)
    
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")


from data_loader import get_loaders  # Assuming get_loaders is a function you've defined to get data loaders

# Initialize data loaders
train_loader, val_loader, test_loader = get_loaders(batch_size)

# Initialize the neural network model and load pre-trained weights
model_path = "63.39_snn_v2.pth"
model_weights = torch.load(model_path)
net = RecurrentSNN_v2(beta, spike_grad, eval_mode=True)
net.load_state_dict(model_weights)
net = net.to(net.device)

# Call the evaluation function
evaluate_and_visualize(net, val_loader, net.device)
