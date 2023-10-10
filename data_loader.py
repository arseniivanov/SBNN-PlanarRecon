import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

