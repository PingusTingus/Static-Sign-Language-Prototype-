import cv2 as cv
import numpy as np
import os
import json
from dynamic_gesture_recognition import DynamicGestureRecognizer
import time

class DataCollector:
    def __init__(self, save_dir='training_data'):
        # Create absolute paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.base_dir, save_dir)
        
        self.recognizer = DynamicGestureRecognizer()
        self.current_gesture = None
        self.recording = False
        self.gesture_data = []
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Load existing gesture mappings if they exist
        self.gesture_mappings_file = os.path.join(self.save_dir, 'gesture_mappings.json')
        if os.path.exists(self.gesture_mappings_file):
            with open(self.gesture_mappings_file, 'r') as f:
                self.gesture_mappings = json.load(f)
        else:
            self.gesture_mappings = {}
            
    def start_recording(self, gesture_name):
        """Start recording a new gesture"""
        self.current_gesture = gesture_name
        self.recording = True
        self.gesture_data = []
        print(f"Recording gesture: {gesture_name}")
        print("Press 'r' to stop recording")
        
    def stop_recording(self):
        """Stop recording the current gesture"""
        if self.recording and self.gesture_data:
            self.recording = False
            # Save the recorded gesture data
            gesture_file = os.path.join(self.save_dir, f"{self.current_gesture}.json")
            with open(gesture_file, 'w') as f:
                json.dump(self.gesture_data, f)
            print(f"Saved {len(self.gesture_data)} frames for gesture: {self.current_gesture}")
            self.gesture_data = []
            
    def collect_data(self):
        """Main data collection loop"""
        cap = cv.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Process frame
            image, gesture_info = self.recognizer.process_frame(frame)
            
            # If recording and we have gesture info, save it
            if self.recording and gesture_info is not None:
                self.gesture_data.append({
                    'features': gesture_info['features'].tolist(),
                    'point': gesture_info['point']
                })
            
            # Draw point history
            for i in range(1, len(self.recognizer.point_history)):
                if i > 0:
                    cv.line(image, self.recognizer.point_history[i - 1], 
                           self.recognizer.point_history[i], (0, 255, 0), 2)
            
            # Display recording status
            if self.recording:
                cv.putText(image, f"Recording: {self.current_gesture}", (10, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the frame
            cv.imshow('Data Collection', image)
            
            # Handle keyboard input
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if self.recording:
                    self.stop_recording()
                else:
                    gesture_name = input("Enter gesture name: ")
                    if gesture_name:
                        self.start_recording(gesture_name)
            
        cap.release()
        cv.destroyAllWindows()

def prepare_training_data(data_dir='training_data'):
    """Prepare the collected data for training"""
    # Create absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, data_dir)
    
    X = []
    y = []
    gesture_mappings = {}
    current_label = 0
    
    # Load all gesture files
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and filename != 'gesture_mappings.json':
            gesture_name = filename[:-5]  # Remove .json extension
            
            # Add to gesture mappings
            if gesture_name not in gesture_mappings:
                gesture_mappings[gesture_name] = current_label
                current_label += 1
            
            # Load gesture data
            with open(os.path.join(data_dir, filename), 'r') as f:
                gesture_data = json.load(f)
                
            # Extract features and labels
            for frame_data in gesture_data:
                X.append(frame_data['features'])
                y.append(gesture_mappings[gesture_name])
    
    # Save gesture mappings
    with open(os.path.join(data_dir, 'gesture_mappings.json'), 'w') as f:
        json.dump(gesture_mappings, f)
    
    return np.array(X), np.array(y), gesture_mappings

def train_model(X, y, save_dir='models'):
    """Train the model and save it"""
    # Create absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize recognizer and train model
    recognizer = DynamicGestureRecognizer()
    model = recognizer.train_model(X, y)
    
    # Save the model
    model_path = os.path.join(save_dir, 'filipino_sign_language_model')
    recognizer.save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

def main():
    # First, collect data
    print("Starting data collection...")
    print("Press 'r' to start/stop recording a gesture")
    print("Press 'q' to quit")
    
    collector = DataCollector()
    collector.collect_data()
    
    # Prepare training data
    print("\nPreparing training data...")
    X, y, gesture_mappings = prepare_training_data()
    print(f"Collected data for {len(gesture_mappings)} gestures")
    print("Gesture mappings:", gesture_mappings)
    
    # Train model
    print("\nTraining model...")
    model = train_model(X, y)
    
    print("\nTraining complete! You can now use the model for real-time recognition.")

if __name__ == '__main__':
    main() 