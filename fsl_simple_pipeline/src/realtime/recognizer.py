import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable
import mediapipe as mp
import os

from ..data_collection.collector import GestureCollector
from ..feature_extraction.extractor import FeatureExtractor
from ..preprocessing.preprocessor import Preprocessor
from ..models.gesture_model import GestureModel

class GestureRecognizer:
    def __init__(self, model_path: str, label_mapping: Dict[int, str], 
                 output_dir: str = "collected_data"):
        """
        Initialize the gesture recognizer.
        
        Args:
            model_path: Path to the trained model
            label_mapping: Mapping from class index to label
            output_dir: Directory to save collected gestures
        """
        self.model_path = model_path
        self.label_mapping = label_mapping
        self.output_dir = output_dir
        
        # Initialize components
        self.collector = GestureCollector(output_dir=output_dir)
        self.feature_extractor = FeatureExtractor()
        self.preprocessor = Preprocessor()
        
        # Load model
        self.model = GestureModel(input_shape=(30, 126), num_classes=len(label_mapping))
        self.model.load(model_path)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # State variables
        self.recording = False
        self.recording_frames = []
        self.last_prediction = None
        self.prediction_history = []
        self.correction_mode = False
        self.correction_frames = []
        
    def recognize_realtime(self, callback: Callable[[str, float], None] = None):
        """
        Perform real-time gesture recognition.
        
        Args:
            callback: Optional callback function to handle predictions
        """
        cap = cv2.VideoCapture(0)
        frame_buffer = []
        buffer_size = 30  # Number of frames to buffer for prediction
        
        print("Real-time gesture recognition started")
        print("Press 'r' to record a new gesture")
        print("Press 'c' to correct the last prediction")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Draw hand landmarks and make predictions
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    
                    # Add to frame buffer
                    frame_buffer.append(landmarks)
                    
                    # Keep buffer size fixed
                    if len(frame_buffer) > buffer_size:
                        frame_buffer.pop(0)
                    
                    # Make prediction if buffer is full
                    if len(frame_buffer) == buffer_size and not self.recording:
                        # Preprocess
                        processed = self.preprocessor.preprocess(np.array(frame_buffer))
                        
                        # Extract features
                        features = self.feature_extractor.extract_features(processed)
                        
                        # Combine features
                        combined_features = np.concatenate([
                            features['handshape'],
                            features['orientation'],
                            features['location'],
                            features['movement']
                        ], axis=1)
                        
                        # Make prediction
                        prediction = self.model.predict(np.expand_dims(combined_features, axis=0))[0]
                        pred_class = np.argmax(prediction)
                        confidence = prediction[pred_class]
                        
                        # Get label
                        label = self.label_mapping[pred_class]
                        
                        # Update state
                        self.last_prediction = (label, confidence)
                        self.prediction_history.append((label, confidence))
                        
                        # Keep history size manageable
                        if len(self.prediction_history) > 10:
                            self.prediction_history.pop(0)
                        
                        # Call callback if provided
                        if callback:
                            callback(label, confidence)
                        
                        # Display prediction
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Handle recording mode
            if self.recording:
                if len(self.recording_frames) > 0:
                    elapsed = time.time() - self.recording_start_time
                    cv2.putText(frame, f"Recording: {elapsed:.1f}s", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Handle correction mode
            if self.correction_mode:
                cv2.putText(frame, "CORRECTION MODE", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Enter correct gesture name", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Real-time Gesture Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not self.recording and not self.correction_mode:
                    # Start recording
                    self.recording = True
                    self.recording_start_time = time.time()
                    self.recording_frames = []
                    print("Recording started...")
                elif self.recording:
                    # Stop recording
                    self.recording = False
                    print(f"Recording stopped. Collected {len(self.recording_frames)} frames.")
                    
                    # Prompt for gesture name
                    gesture_name = input("Enter gesture name: ")
                    if gesture_name:
                        # Save the gesture
                        self.collector.save_gesture(gesture_name, np.array(self.recording_frames))
                        print(f"Saved gesture as '{gesture_name}'")
            elif key == ord('c'):
                if not self.recording and self.last_prediction is not None:
                    # Enter correction mode
                    self.correction_mode = True
                    self.correction_frames = []
                    print("Correction mode activated")
                    print(f"Last prediction: {self.last_prediction[0]} ({self.last_prediction[1]:.2f})")
                    
                    # Prompt for correct gesture name
                    correct_name = input("Enter correct gesture name: ")
                    if correct_name:
                        # Save the correction
                        self.collector.save_gesture(correct_name, np.array(frame_buffer))
                        print(f"Saved correction as '{correct_name}'")
                        self.correction_mode = False
        
        cap.release()
        cv2.destroyAllWindows()
    
    def add_new_gesture(self, gesture_name: str):
        """
        Add a new gesture to the dataset.
        
        Args:
            gesture_name: Name of the new gesture
        """
        print(f"Collecting new gesture: {gesture_name}")
        landmarks = self.collector.collect_gesture_realtime()
        
        if landmarks is not None:
            self.collector.save_gesture(gesture_name, landmarks)
            print(f"Added new gesture: {gesture_name}")
            return True
        
        return False
    
    def correct_prediction(self, correct_gesture: str):
        """
        Correct a misclassified gesture.
        
        Args:
            correct_gesture: Correct gesture name
        """
        print(f"Collecting correction for: {correct_gesture}")
        landmarks = self.collector.collect_gesture_realtime()
        
        if landmarks is not None:
            self.collector.save_gesture(correct_gesture, landmarks)
            print(f"Saved correction as: {correct_gesture}")
            return True
        
        return False 