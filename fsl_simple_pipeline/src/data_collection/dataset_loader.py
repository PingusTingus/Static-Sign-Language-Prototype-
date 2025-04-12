import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import mediapipe as mp

class DatasetLoader:
    def __init__(self, dataset_path: str = "FSL SLR Dataset"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the FSL SLR Dataset
        """
        self.dataset_path = dataset_path
        self.labels_df = pd.read_csv(os.path.join(dataset_path, "labels.csv"))
        self.train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels_df['label'].unique())}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def load_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Load a video and extract hand landmarks.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Array of hand landmarks if successful, None otherwise
        """
        full_path = os.path.join(self.dataset_path, video_path)
        if not os.path.exists(full_path):
            print(f"Video not found: {full_path}")
            return None
            
        cap = cv2.VideoCapture(full_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Extract landmarks if hands are detected
            if results.multi_hand_landmarks:
                # Use the first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                frames.append(landmarks)
        
        cap.release()
        
        if not frames:
            return None
            
        return np.array(frames)
    
    def load_dataset(self, split: str = "train", max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the dataset.
        
        Args:
            split: Dataset split to load ("train" or "test")
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            Tuple of (features, labels)
        """
        if split == "train":
            df = self.train_df
        elif split == "test":
            df = self.test_df
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if max_samples is not None:
            df = df.sample(min(max_samples, len(df)))
        
        X = []
        y = []
        
        for _, row in df.iterrows():
            video_path = row['vid_path']
            label = row['label']
            label_idx = self.label_to_idx[label]
            
            landmarks = self.load_video(video_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label_idx)
        
        return np.array(X), np.array(y)
    
    def get_label_mapping(self) -> Dict[str, int]:
        """Get the label to index mapping."""
        return self.label_to_idx
    
    def get_index_mapping(self) -> Dict[int, str]:
        """Get the index to label mapping."""
        return self.idx_to_label 