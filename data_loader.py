# Import necessary libraries for creating the dataloader
import numpy as np
import torch
from torch.utils.data import Dataset

# Define the Moving MNIST dataset class
class MovingMNIST(Dataset):
    def __init__(self, npy_file_path):
        # Load the .npy file
        self.data = np.load(npy_file_path)
        
        # Move the data to channel-last format to channel-first format
        # Original shape: (num_samples, sequence_length, height, width)
        # New shape: (num_samples, sequence_length, 1, height, width)
        self.data = np.expand_dims(self.data, axis=2)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Fetch the sequence at the given index
        sequence = self.data[idx]
        
        # Convert the sequence to PyTorch tensor
        sequence_tensor = torch.FloatTensor(sequence)
        
        # For object classification, you may want to return the first 10 frames as input and the last 10 frames as target
        input_tensor = sequence_tensor[:10]
        target_tensor = sequence_tensor[10:]
        
        return input_tensor, target_tensor

def rgb_to_yuv(img):
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            img = img.permute(0, 2, 3, 1)  # Change (B,C,H,W) to (B,H,W,C) for pytorch tensor
        else:
            img = img.permute(1, 2, 0)  # Change (C,H,W) to (H,W,C) for pytorch tensor
    img = img.to(torch.float32)
    in_img = img.contiguous().view(-1, 3)
    out_img = torch.mm(in_img, torch.tensor([[0.299, -0.14713, 0.615],
                                             [0.587, -0.28886, -0.51499],
                                             [0.114, 0.436, -0.10001]]).t()).view(*img.shape)
    return out_img