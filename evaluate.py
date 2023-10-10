from data_loader import get_loaders

import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

from config import num_steps, batch_size, num_classes, beta, spike_grad
if spike_grad == "fast_sigmoid":
    from snntorch import surrogate
    spike_grad = surrogate.fast_sigmoid()

train_loader, val_loader, test_loader = get_loaders(batch_size)


# Example of how to iterate through DataLoader
for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape, target)

net = nn.Sequential(
    nn.Conv2d(1, 16, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
    nn.Conv2d(16, 32, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
    nn.Flatten(),
    nn.Linear(32 * 28 * 38, 256),  # Adjusted size
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
    nn.Linear(256, num_classes),
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)
)

# Existing code for DataLoader and SNN definition here...

# Initialize spike recordings
spike_recordings = []

# Reset/initialize hidden states for all neurons
utils.reset(net)

# Iterate through DataLoader
for batch_idx, (data, target) in enumerate(train_loader):
    # Reshape data to be compatible with the SNN
    data = data.permute(1, 0, 2, 3, 4)
    
    # Reset spike recordings for new batch
    batch_spike_recordings = []
    
    # Forward pass through time
    for step in range(num_steps):
        spike, state = net(data[step])
        batch_spike_recordings.append(spike)
    
    spike_recordings.append(torch.stack(batch_spike_recordings))

    # Your training logic here...
