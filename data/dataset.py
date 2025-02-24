import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VideoFrameDataset(Dataset):
    def __init__(self, video_dir, transform=None, keyframe_interval=5):
        """
        video_dir: directory containing video files or extracted frames.
        keyframe_interval: how many frames to skip between keyframes.
        """
        self.video_dir = video_dir
        self.transform = transform
        # Assume each video is a subfolder with frame images
        self.video_folders = [os.path.join(video_dir, folder) for folder in os.listdir(video_dir)]
        self.keyframe_interval = keyframe_interval

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_path = self.video_folders[idx]
        frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg') or f.endswith('.png')])
        
        # Load keyframes (e.g., first and the keyframe_interval-th frame)
        keyframes = []
        for i in range(0, len(frame_files), self.keyframe_interval):
            frame = cv2.imread(frame_files[i])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = transforms.ToTensor()(frame)
            keyframes.append(frame)
        
        # For simplicity, we assume two keyframes per sample for interpolation
        if len(keyframes) >= 2:
            keyframe1 = keyframes[0]
            keyframe2 = keyframes[1]
        else:
            keyframe1 = keyframes[0]
            keyframe2 = keyframes[0]

        return keyframe1, keyframe2

if __name__ == '__main__':
    from torchvision.transforms import Compose, Resize, ToTensor
    transform = Compose([Resize((64, 64)), ToTensor()])
    # dataset = VideoFrameDataset(video_dir='data/videos', transform=transform)
    dataset = VideoFrameDataset(video_dir='/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/4', transform=transform)
    keyframe1, keyframe2 = dataset[0]
    print("Keyframe shapes:", keyframe1.shape, keyframe2.shape)
