import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_loader import KTHDataset


# Load video paths and labels
actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
video_files = []
labels = []

for i, action in enumerate(actions):
    folder = os.path.join("data/KTH_Dataset", action)
    for filename in os.listdir(folder):
        if filename.endswith(".avi"):
            path = os.path.join(folder, filename)
            video_files.append(path)
            labels.append(i)

# Split data into training, validation, and test sets
videos_train, videos_test, labels_train, labels_test = train_test_split(video_files, labels, test_size=0.2, random_state=42, stratify=labels)
videos_train, videos_val, labels_train, labels_val = train_test_split(videos_train, labels_train, test_size=0.25, random_state=42, stratify=labels_train)

train_data = [(v, l) for v, l in zip(videos_train, labels_train)]
val_data = [(v, l) for v, l in zip(videos_val, labels_val)]
test_data = [(v, l) for v, l in zip(videos_test, labels_test)]

train_dataset = KTHDataset(train_data)
val_dataset = KTHDataset(val_data)
test_dataset = KTHDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

# Example of how to iterate through DataLoader
for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape, target)