import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import num_steps

class KTHDataset(Dataset):
    def __init__(self, video_list, transform=None, frame_skip=5, use_diff=True):
        self.video_list = video_list
        self.transform = transform
        self.FIXED_NUM_FRAMES = num_steps
        self.frame_skip = frame_skip
        self.use_diff = use_diff

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        path, label = self.video_list[idx]
        cap = cv2.VideoCapture(path)
        frames = []
        prev_frame = None
        frame_count = 0

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                if frame_count % self.frame_skip == 0:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if self.use_diff:
                        if prev_frame is not None:
                            diff_frame = cv2.absdiff(gray_frame, prev_frame)
                            norm_diff_frame = diff_frame / 255.0
                            frames.append(norm_diff_frame)
                        prev_frame = gray_frame
                    else:
                        frames.append(gray_frame)

                frame_count += 1
            else:
                break

        cap.release()

        frames = np.array(frames)

        # Zero pad to ensure all video samples have FIXED_NUM_FRAMES
        if len(frames) < self.FIXED_NUM_FRAMES:
            pad_len = self.FIXED_NUM_FRAMES - len(frames)
            zero_pad = np.zeros((pad_len, 120, 160), dtype=np.uint8 if not self.use_diff else np.float32)
            frames = np.concatenate((frames, zero_pad), axis=0)
        elif len(frames) > self.FIXED_NUM_FRAMES:
            frames = frames[:self.FIXED_NUM_FRAMES]

        if self.transform:
            frames = self.transform(frames)

        return torch.tensor(frames, dtype=torch.float32).unsqueeze(1), torch.tensor(label, dtype=torch.long)

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

    train_dataset = KTHDataset(train_data, use_diff=True)
    val_dataset = KTHDataset(val_data, use_diff=True)
    test_dataset = KTHDataset(test_data, use_diff=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_loader, val_loader, test_loader
