import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt


from config import num_steps, num_classes, DEBUG

def k_wta(x, sparsity_level):
    k = int(sparsity_level * x.shape[1])
    topk, _ = x.topk(k, dim=1)
    threshold = topk[:, -1].unsqueeze(1)
    return F.relu(x - threshold)

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

        self.dropout = nn.Dropout(0.3)
        self.device = "cpu"

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

class RecurrentSNN_v2(nn.Module):
    def __init__(self, beta, spike_grad, eval_mode=False, plot_pred=False):
        super(RecurrentSNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.lstm1 = nn.LSTM(16, 16) #Local recurrent connections
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(7488, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.leaky1 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky2 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky3 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky4 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky_out = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)

        self.attention = nn.Linear(32, 32)  # Assuming 64 channels in spike3
        self.dropout = nn.Dropout(0.3)
        self.device = "cpu"
        self.decay_factor = 0.7
        self.eval_mode = eval_mode
        self.plot_pred = plot_pred

    def forward(self, x, target_label=None):
        recurrent_spike = None

        if self.eval_mode:
            self.eval_data = x.cpu().detach().numpy()

        for step in range(num_steps):

            out = self.conv1(x[step])
            out = self.bn1(out)

            spike1 = self.leaky1(out)
            spike1 = k_wta(spike1, sparsity_level=0.6)
            if DEBUG:
                print(f"Layer 1, Step {step}, Spike Sum: {spike1.sum().item()}")

            if recurrent_spike is not None:
                spike1 = spike1 + recurrent_spike  # Add recurrent connection
        
            out = self.conv2(spike1)
            out = self.bn2(out)
            spike2 = self.leaky2(out)
            if DEBUG:
                print(f"Layer 2, Step {step}, Spike Sum: {spike2.sum().item()}")

            out = self.conv3(spike2)
            out = self.bn3(out)

            spike3 = self.leaky3(out)
            out = spike3
            if DEBUG:
                print(f"Layer 3, Step {step}, Spike Sum: {spike3.sum().item()}")

            attention_weights = F.softmax(self.attention(spike3.permute(0, 2, 3, 1)), dim=3)
            out = out * attention_weights.permute(0, 3, 1, 2)

            out = nn.Flatten()(out)
            out = self.fc1(out)
            spike4 = self.leaky4(out)
            if DEBUG:
                print(f"Layer 4, Step {step}, Spike Sum: {spike4.sum().item()}")

            out = self.fc2(spike4)
            spike_out, _ = self.leaky_out(out)
            if DEBUG:
                print(f"Output Layer, Step {step}, Spike Sum: {spike_out.sum().item()}")

            recurrent_spike = self.dropout(spike1)   # Store spike for recurrent connection
            recurrent_spike *= self.decay_factor

            if self.eval_mode and self.plot_pred:
                pred_label = spike_out.argmax(dim=1).cpu().numpy()
                img_to_show = self.eval_data[step, 0, 0, :, :]
                plt.imshow(img_to_show, cmap='gray')
                title_str = f"Step: {step}, Prediction: {pred_label}"
                if target_label is not None:
                    title_str += f", Target: {target_label}"
                plt.title(title_str)
                plt.pause(0.1)
                plt.clf()

        return spike_out

class RecurrentSNN_v3(nn.Module):
    def __init__(self, beta, spike_grad, eval_mode=False, plot_pred=False):
        super(RecurrentSNN_v3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(7488, 256)
        self.fc2 = nn.Linear(256, 192)
        self.fc3 = nn.Linear(192, num_classes)

        self.leaky1 = snn.Leaky(beta=0.5, init_hidden=True, spike_grad=spike_grad)
        self.leaky2 = snn.Leaky(beta=0.6, init_hidden=True, spike_grad=spike_grad)
        self.leaky3 = snn.Leaky(beta=0.7, init_hidden=True, spike_grad=spike_grad)
        self.leaky4 = snn.Leaky(beta=0.7, init_hidden=True, spike_grad=spike_grad)
        self.leaky5 = snn.Leaky(beta=0.75, init_hidden=True, spike_grad=spike_grad)
        self.leaky6 = snn.Leaky(beta=0.85, init_hidden=True, spike_grad=spike_grad)
        self.leaky_out = snn.Leaky(beta=0.9, init_hidden=True, spike_grad=spike_grad, output=True)

        self.attention = nn.Linear(32, 32)  # Assuming 64 channels in spike3
        self.dropout = nn.Dropout(0.3)
        self.device = "cpu"
        self.decay_factor = 0.7
        self.eval_mode = eval_mode
        self.plot_pred = plot_pred
            
        self.horizontal_cell = nn.Conv2d(1, 16, 11, stride=2, padding=3)
        self.bn_horizontal = nn.BatchNorm2d(16)
        self.leaky_horizontal = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1d_q = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1)
        self.conv1d_k = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1)

        # Global Spatial Attention
        self.conv_q_spatial = nn.Conv2d(16, 16, kernel_size=1)
        self.conv_k_spatial = nn.Conv2d(16, 16, kernel_size=1)
        self.conv_v_spatial = nn.Conv2d(16, 16, kernel_size=1)

    def forward(self, x, target_label=None):
        recurrent_spikes = {}

        if self.eval_mode:
            self.eval_data = x.cpu().detach().numpy()

        for step in range(num_steps):

            out = self.conv1(x[step])
            out = self.bn1(out)

            spike1 = self.leaky1(out)
            spike1 = k_wta(spike1, sparsity_level=0.6)
            recurrent_spikes[1] = spike1 * 0.9

            if DEBUG:
                print(f"Layer 1, Step {step}, Spike Sum: {spike1.sum().item()}")

            if step > 0:
                combined_spike1 = spike1 + recurrent_spikes[1]
        
            out = self.conv2(combined_spike1)
            out = self.bn2(out)
            spike2 = self.leaky2(out)

            recurrent_spikes[2] = spike2 * 0.8
            if step > 0:
                spike2 = spike2 + recurrent_spikes[2]

            if DEBUG:
                print(f"Layer 2, Step {step}, Spike Sum: {spike2.sum().item()}")

            out = self.conv3(spike2)
            out = self.bn3(out)
            spike3 = self.leaky3(out)
            recurrent_spikes[3] = spike3 * 0.7
            if step > 0:
                spike3 = spike3 + recurrent_spikes[3]
            if DEBUG:
                print(f"Layer 3, Step {step}, Spike Sum: {spike3.sum().item()}")

            #out = self.conv4(spike3)
            #out = self.bn4(out)
            out = nn.Flatten()(spike3)
            out = self.fc1(out)
            spike4 = self.leaky4(out)
            recurrent_spikes[4] = spike4 * 0.6
            if step > 0:
                spike4 = spike4 + recurrent_spikes[4]

            if DEBUG:
                print(f"Layer 4, Step {step}, Spike Sum: {spike4.sum().item()}")

            #out = self.conv5(spike4)
            #out = self.bn5(out)
            out = self.fc2(spike4)
            spike5 = self.leaky5(out)
            out = spike5
            recurrent_spikes[5] = spike5 * 0.5
            if step > 0:
                spike5 = spike5 + recurrent_spikes[5]

            if DEBUG:
                print(f"Layer 5, Step {step}, Spike Sum: {spike5.sum().item()}")

            #out = nn.Flatten()(out)
            #out = self.fc1(out)
            #spike6 = self.leaky6(out)
            #if DEBUG:
            #    print(f"Layer 6, Step {step}, Spike Sum: {spike6.sum().item()}")

            out = self.fc3(out)
            spike_out, _ = self.leaky_out(out)
            if DEBUG:
                print(f"Output Layer, Step {step}, Spike Sum: {spike_out.sum().item()}")

            if self.eval_mode and self.plot_pred:
                pred_label = spike_out.argmax(dim=1).cpu().numpy()
                img_to_show = self.eval_data[step, 0, 0, :, :]
                plt.imshow(img_to_show, cmap='gray')
                title_str = f"Step: {step}, Prediction: {pred_label}"
                if target_label is not None:
                    title_str += f", Target: {target_label}"
                plt.title(title_str)
                plt.pause(0.1)
                plt.clf()

        return spike_out
# Adjusting the architecture to use strides for dimensionality reduction in convolutional layers

class RecurrentSNN_v4(nn.Module):
    def __init__(self, beta, spike_grad, eval_mode=False, plot_pred=False):
        super(RecurrentSNN_v4, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(7488, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.leaky1 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky2 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky3 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky4 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky_out = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)

        self.attention = nn.Linear(32, 32)  # Assuming 64 channels in spike3
        self.dropout = nn.Dropout(0.3)
        self.device = "cpu"
        self.decay_factor = 0.7
        self.eval_mode = eval_mode
        self.plot_pred = plot_pred

    def forward(self, x, target_label=None):
        recurrent_spike = None

        if self.eval_mode:
            self.eval_data = x.cpu().detach().numpy()

        for step in range(x.shape[0]):

            out = self.conv1(x[step])
            out = self.bn1(out)

            spike1 = self.leaky1(out)
            spike1 = k_wta(spike1, sparsity_level=0.6)
            if DEBUG:
                print(f"Layer 1, Step {step}, Spike Sum: {spike1.sum().item()}")

            if recurrent_spike is not None:
                spike1 = spike1 + recurrent_spike  # Add recurrent connection
        
            out = self.conv2(spike1)
            out = self.bn2(out)
            spike2 = self.leaky2(out)
            if DEBUG:
                print(f"Layer 2, Step {step}, Spike Sum: {spike2.sum().item()}")

            out = self.conv3(spike2)
            out = self.bn3(out)

            spike3 = self.leaky3(out)
            out = spike3
            if DEBUG:
                print(f"Layer 3, Step {step}, Spike Sum: {spike3.sum().item()}")

            attention_weights = F.softmax(self.attention(spike3.permute(0, 2, 3, 1)), dim=3)
            out = out * attention_weights.permute(0, 3, 1, 2)

            out = nn.Flatten()(out)
            out = self.fc1(out)
            spike4 = self.leaky4(out)
            if DEBUG:
                print(f"Layer 4, Step {step}, Spike Sum: {spike4.sum().item()}")

            out = self.fc2(spike4)
            spike_out, _ = self.leaky_out(out)
            if DEBUG:
                print(f"Output Layer, Step {step}, Spike Sum: {spike_out.sum().item()}")

            recurrent_spike = self.dropout(spike1)   # Store spike for recurrent connection
            recurrent_spike *= self.decay_factor

            if self.eval_mode and self.plot_pred:
                pred_label = spike_out.argmax(dim=1).cpu().numpy()
                img_to_show = self.eval_data[step, 0, 0, :, :]
                plt.imshow(img_to_show, cmap='gray')
                title_str = f"Step: {step}, Prediction: {pred_label}"
                if target_label is not None:
                    title_str += f", Target: {target_label}"
                plt.title(title_str)
                plt.pause(0.1)
                plt.clf()

        return spike_out

class RecurrentSNN_v5(nn.Module):
    def __init__(self, beta, spike_grad, eval_mode=False, plot_pred=False):
        super(RecurrentSNN_v5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.fc1 = nn.Linear(7488, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.leaky1 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky2 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky3 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky4 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
        self.leaky_out = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)

        self.attention = nn.Linear(32, 32)  # Assuming 64 channels in spike3
        self.dropout = nn.Dropout(0.3)
        self.device = "cpu"
        self.decay_factor = 0.7
        self.eval_mode = eval_mode
        self.plot_pred = plot_pred

    def forward(self, x, target_label=None):
        recurrent_spike = None

        if self.eval_mode:
            self.eval_data = x.cpu().detach().numpy()

        for step in range(num_steps):

            out = self.conv1(x[step])

            spike1 = self.leaky1(out)
            spike1 = k_wta(spike1, sparsity_level=0.6)
            if DEBUG:
                print(f"Layer 1, Step {step}, Spike Sum: {spike1.sum().item()}")

            if recurrent_spike is not None:
                spike1 = spike1 + recurrent_spike  # Add recurrent connection
        
            out = self.conv2(spike1)
            spike2 = self.leaky2(out)
            if DEBUG:
                print(f"Layer 2, Step {step}, Spike Sum: {spike2.sum().item()}")

            out = self.conv3(spike2)

            spike3 = self.leaky3(out)
            out = spike3
            if DEBUG:
                print(f"Layer 3, Step {step}, Spike Sum: {spike3.sum().item()}")

            attention_weights = F.softmax(self.attention(spike3.permute(0, 2, 3, 1)), dim=3)
            out = out * attention_weights.permute(0, 3, 1, 2)

            out = nn.Flatten()(out)
            out = self.fc1(out)
            spike4 = self.leaky4(out)
            if DEBUG:
                print(f"Layer 4, Step {step}, Spike Sum: {spike4.sum().item()}")

            out = self.fc2(spike4)
            spike_out, _ = self.leaky_out(out)
            if DEBUG:
                print(f"Output Layer, Step {step}, Spike Sum: {spike_out.sum().item()}")

            recurrent_spike = self.dropout(spike1)   # Store spike for recurrent connection
            recurrent_spike *= self.decay_factor

            if self.eval_mode and self.plot_pred:
                pred_label = spike_out.argmax(dim=1).cpu().numpy()
                img_to_show = self.eval_data[step, 0, 0, :, :]
                plt.imshow(img_to_show, cmap='gray')
                title_str = f"Step: {step}, Prediction: {pred_label}"
                if target_label is not None:
                    title_str += f", Target: {target_label}"
                plt.title(title_str)
                plt.pause(0.1)
                plt.clf()

        return spike_out

class CorticalColumnNetV4(nn.Module):
    def __init__(self):
        super(CorticalColumnNetV4, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        
        # Using GRU instead of LSTM
        self.gru = nn.GRU(32 * 30 * 40, 128, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc_out = nn.Linear(32, num_classes)
        self.device = "cuda"
        
    def forward(self, x):
        timesteps, batch_size, C, H, W = x.size()
        
        c_in = x.permute(1, 0, 2, 3, 4).contiguous().view(-1, C, H, W)
        
        c_out = F.relu(self.conv1(c_in))
        c_out = F.relu(self.conv2(c_out))
        
        r_in = c_out.view(batch_size, timesteps, -1)
        
        # Using GRU
        r_out, hn = self.gru(r_in)
        r_out_last = r_out[:, -1, :]
        
        f_out = F.relu(self.fc1(r_out_last))
        f_out = self.dropout1(f_out)
        f_out = F.relu(self.fc2(f_out))
        f_out = self.dropout2(f_out)
        
        out = self.fc_out(f_out)
        
        return out