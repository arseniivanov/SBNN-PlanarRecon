import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class KTHDataset(Dataset):
    def __init__(self, video_list, frame_number=400,transform=None):
        self.video_list = video_list
        self.transform = transform
        self.FIXED_NUM_FRAMES = frame_number

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        path, label = self.video_list[idx]
        cap = cv2.VideoCapture(path)
        frames = []

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray_frame)
            else:
                break

        cap.release()

        # Normalize the number of frames to FIXED_NUM_FRAMES
        if len(frames) > self.FIXED_NUM_FRAMES:
            frames = frames[:self.FIXED_NUM_FRAMES]  # Truncate
        elif len(frames) < self.FIXED_NUM_FRAMES:
            # Zero-pad
            while len(frames) < self.FIXED_NUM_FRAMES:
                frames.append(np.zeros((120, 160), dtype=np.uint8))

        frames = np.array(frames)
        if self.transform:
            frames = self.transform(frames)

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Your existing code for KTHDataset class here...

def get_loaders(batch_size=4):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
