num_steps = 80  # This should be the same as FIXED_NUM_FRAMES
batch_size = 4  # Adjust as per your hardware capabilities
num_classes = 6  # For the 6 action categories in KTH dataset
beta = 0.5  # Neuron decay rate
spike_grad = "fast_sigmoid" # Surrogate gradient