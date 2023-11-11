import json
import matplotlib.pyplot as plt

# Load the data from the file
data_lengths_path = 'data_lengths.json'
with open(data_lengths_path, 'r') as file:
    data_lengths = json.load(file)

# Extract frame lengths
frame_lengths = list(data_lengths.values())

# Create a histogram
plt.hist(frame_lengths, bins=20, color='blue', edgecolor='black')
plt.xlabel('Frame Length')
plt.ylabel('Frequency')
plt.title('Histogram of Frame Lengths')

# Save the plot
histogram_path = 'frame_length_stats.jpg'
plt.savefig(histogram_path)

# Actions for classification
actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']

# Calculate mean video length for each class
mean_lengths = {}
for action in actions:
    # Extract frame lengths for this action
    action_lengths = [length for key, length in data_lengths.items() if action in key]
    
    # Calculate and store the mean length
    mean_lengths[action] = sum(action_lengths) / len(action_lengths)

print(mean_lengths)

