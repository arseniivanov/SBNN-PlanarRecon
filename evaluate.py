from data_loader import get_loaders

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

import torch.optim as optim
from sklearn.metrics import accuracy_score

from config import num_steps, batch_size, num_classes, beta, spike_grad
if spike_grad == "fast_sigmoid":
    spike_grad = surrogate.fast_sigmoid()

train_loader, val_loader, test_loader = get_loaders(batch_size)

net = nn.Sequential(
    nn.Conv2d(1, 16, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
    nn.Conv2d(16, 32, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
    nn.Flatten(),
    nn.Linear(31968, 256),  # Adjusted size
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
    nn.Linear(256, num_classes),
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Number of epochs
num_epochs = 10

for epoch in range(num_epochs):
    net.train()
    epoch_train_loss = 0
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Reset/initialize hidden states for all neurons
        utils.reset(net)

        data = data.permute(1, 0, 2, 3, 4)  # Permute for time steps

        # Forward pass through time
        for step in range(num_steps):
            spike, state = net(data[step])
        
        loss = criterion(spike, target)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
    
    # Validation loop
    net.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            # Reset/initialize hidden states for all neurons
            utils.reset(net)
            
            data = data.unsqueeze(2)  # Add channel dimension
            data = data.permute(1, 0, 2, 3, 4)  # Permute for time steps

            # Forward pass through time
            for step in range(num_steps):
                spike, state = net(data[step])
            
            all_preds.append(spike.argmax(dim=1).cpu().numpy())
            all_labels.append(target.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    val_accuracy = accuracy_score(all_preds, all_labels)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy*100:.2f}%")
