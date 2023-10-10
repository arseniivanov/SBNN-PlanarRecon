from data_loader import MovingMNIST
from torch.utils.data import DataLoader

# Create an instance of the dataset
moving_mnist_dataset = MovingMNIST("data/moving_mnist.npy")

# Create a dataloader
dataloader = DataLoader(moving_mnist_dataset, batch_size=32, shuffle=True)

# Iterate through the dataloader
for inputs, targets in dataloader:
   print(inputs.shape, targets.shape)
