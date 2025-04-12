import cv2
import mediapipe as mp
import numpy as np
import os
import time
from typing import List, Tuple, Optional, Dict, Callable

class GestureCollector:
    def __init__(self, output_dir: str = "collected_data"):
        """Initialize the gesture collector."""
        self.output_dir = output_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def collect_gesture(self, gesture_name: str, num_frames: int = 30) -> Optional[np.ndarray]:
        """
        Collect a single gesture sequence.
        
        Args:
            gesture_name: Name of the gesture to collect
            num_frames: Number of frames to collect
            
        Returns:
            Array of hand landmarks if successful, None otherwise
        """
        cap = cv2.VideoCapture(0)
        frames = []
        frame_count = 0
        
        print(f"Collecting gesture: {gesture_name}")
        print("Press 'q' to quit, 's' to start collecting")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    frames.append(landmarks)
                    frame_count += 1
            
            # Display frame
            cv2.imshow('Gesture Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and frame_count >= num_frames:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(frames) >= num_frames:
            # Save the gesture data
            gesture_data = np.array(frames)
            gesture_dir = os.path.join(self.output_dir, gesture_name)
            os.makedirs(gesture_dir, exist_ok=True)
            
            # Save as numpy array
            save_path = os.path.join(gesture_dir, f"{len(os.listdir(gesture_dir))}.npy")
            np.save(save_path, gesture_data)
            print(f"Saved gesture data to {save_path}")
            return gesture_data
        
        return None
    
    def collect_gesture_realtime(self, callback: Callable[[np.ndarray], None] = None) -> Optional[np.ndarray]:
        """
        Collect a gesture in real-time with callback for processing.
        
        Args:
            callback: Optional callback function to process landmarks in real-time
            
        Returns:
            Array of hand landmarks if successful, None otherwise
        """
        cap = cv2.VideoCapture(0)
        frames = []
        frame_count = 0
        recording = False
        start_time = None
        
        print("Press 'r' to start/stop recording, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    
                    # If recording, save landmarks
                    if recording:
                        frames.append(landmarks)
                        frame_count += 1
                        
                        # Call callback if provided
                        if callback:
                            callback(landmarks)
            
            # Display recording status
            if recording:
                elapsed = time.time() - start_time
                cv2.putText(frame, f"Recording: {elapsed:.1f}s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Press 'r' to record", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Real-time Gesture Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    # Start recording
                    recording = True
                    start_time = time.time()
                    frames = []
                    frame_count = 0
                    print("Recording started...")
                else:
                    # Stop recording
                    recording = False
                    print(f"Recording stopped. Collected {frame_count} frames.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            return np.array(frames)
        
        return None
    
    def save_gesture(self, gesture_name: str, landmarks: np.ndarray) -> str:
        """
        Save a gesture to disk.
        
        Args:
            gesture_name: Name of the gesture
            landmarks: Array of hand landmarks
            
        Returns:
            Path to the saved file
        """
        gesture_dir = os.path.join(self.output_dir, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)
        
        # Save as numpy array
        save_path = os.path.join(gesture_dir, f"{len(os.listdir(gesture_dir))}.npy")
        np.save(save_path, landmarks)
        print(f"Saved gesture data to {save_path}")
        
        return save_path 