from data_loader import get_loaders

import torch.nn as nn
import snntorch as snn

from config import num_steps, num_classes, DEBUG

class RecurrentSNN(nn.Module):
    def __init__(self, beta, spike_grad):
        super(RecurrentSNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.fc1 = nn.Linear(7488, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.leaky1 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky2 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky3 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky4 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky_out = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        recurrent_spike = None

        for step in range(num_steps):
            out = self.conv1(x[step])
            spike1 = self.leaky1(out)
            if DEBUG:
                print(f"Layer 1, Step {step}, Spike Sum: {spike1.sum().item()}")

            if recurrent_spike is not None:
                spike1 += recurrent_spike  # Add recurrent connection

            out = self.conv2(spike1)
            spike2 = self.leaky2(out)
            if DEBUG:
                print(f"Layer 2, Step {step}, Spike Sum: {spike2.sum().item()}")

            out = self.conv3(spike2)
            spike3 = self.leaky3(out)
            if DEBUG:
                print(f"Layer 3, Step {step}, Spike Sum: {spike3.sum().item()}")

            out = nn.Flatten()(spike3)
            out = self.fc1(out)
            spike4 = self.leaky4(out)
            if DEBUG:
                print(f"Layer 4, Step {step}, Spike Sum: {spike4.sum().item()}")

            out = self.fc2(spike4)
            spike_out, _ = self.leaky_out(out)
            if DEBUG:
                print(f"Output Layer, Step {step}, Spike Sum: {spike_out.sum().item()}")

            recurrent_spike = self.dropout(spike1)  # Store spike for recurrent connection

        return spike_out
