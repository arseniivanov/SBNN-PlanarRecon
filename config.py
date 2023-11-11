num_steps = 60  # This should be the same as FIXED_NUM_FRAMES
batch_size = 1  # Adjust as per your hardware capabilities
num_classes = 6  # For the 6 action categories in KTH dataset
beta = 0.7  # Neuron decay rate
spike_grad = "fast_sigmoid" # Surrogate gradient
DEBUG = False
net = "RecurrentSNN"