import cv2 as cv
import numpy as np
import os
import json
from dynamic_gesture_recognition import DynamicGestureRecognizer

class DataCollector:
    def __init__(self, save_dir='training_data'):
        # Create absolute paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.base_dir, save_dir)
        
        self.recognizer = DynamicGestureRecognizer()
        self.current_gesture = None
        self.recording = False
        self.gesture_data = []
        self.no_hands_counter = 0  # Counter for frames with no hands
        self.max_no_hands_frames = 30  # Stop recording after 30 frames (1 second) of no hands
        self.target_repetitions = 0  # Number of repetitions to collect
        self.current_repetition = 0  # Current repetition count
        self.waiting_for_next = False  # Flag to indicate waiting for next repetition
        self.wait_frames = 0  # Counter for waiting frames
        self.wait_duration = 60  # Wait for 2 seconds (60 frames) before next repetition
        
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
            
    def start_recording(self, gesture_name, repetitions=1):
        """Start recording a new gesture"""
        self.current_gesture = gesture_name
        self.recording = True
        self.gesture_data = []
        self.no_hands_counter = 0
        self.target_repetitions = repetitions
        self.current_repetition = 0
        self.waiting_for_next = False
        self.wait_frames = 0
        print(f"Recording gesture: {gesture_name}")
        print(f"Target repetitions: {repetitions}")
        print("Perform the gesture. Recording will stop automatically when no hands are detected for 1 second.")
        print("Press 'q' to quit")
        
    def stop_recording(self):
        """Stop recording the current gesture"""
        if self.recording and self.gesture_data:
            self.recording = False
            self.current_repetition += 1
            
            # Save the recorded gesture data with repetition number
            gesture_file = os.path.join(self.save_dir, f"{self.current_gesture}_{self.current_repetition}.json")
            
            # Save this repetition as a separate sample
            with open(gesture_file, 'w') as f:
                json.dump(self.gesture_data, f)
            
            print(f"Saved repetition {self.current_repetition}/{self.target_repetitions} for gesture: {self.current_gesture}")
            print(f"Frames in this repetition: {len(self.gesture_data)}")
            
            # Clear gesture data for next repetition
            self.gesture_data = []
            
            # Check if we need to continue recording repetitions
            if self.current_repetition < self.target_repetitions:
                print(f"\nGet ready for repetition {self.current_repetition + 1}/{self.target_repetitions}")
                print("Next repetition will start automatically in 2 seconds...")
                self.waiting_for_next = True
                self.wait_frames = 0
            else:
                print(f"\nCompleted all {self.target_repetitions} repetitions for {self.current_gesture}")

    def collect_data(self):
        """Main data collection loop"""
        try:
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera. Please check your camera connection.")
                return

            print("Camera initialized successfully.")
            print("Press 'r' to start recording a gesture")
            print("Press 'q' to quit")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Error: Failed to read frame from camera. Retrying...")
                    # Try to reinitialize the camera
                    cap.release()
                    cap = cv.VideoCapture(0)
                    if not cap.isOpened():
                        print("Error: Could not reopen camera. Please check your camera connection.")
                        break
                    continue

                # Process frame
                image, gesture_info = self.recognizer.process_frame(frame)

                # Handle waiting period between repetitions
                if self.waiting_for_next:
                    self.wait_frames += 1
                    if self.wait_frames >= self.wait_duration:
                        self.waiting_for_next = False
                        self.recording = True
                        self.no_hands_counter = 0
                        print(f"Starting repetition {self.current_repetition + 1}/{self.target_repetitions}")

                # If recording and we have gesture info, save it
                if self.recording:
                    if gesture_info is not None:
                        self.gesture_data.append({
                            'features': gesture_info['features'].tolist(),
                            'point': gesture_info['point']
                        })
                        self.no_hands_counter = 0  # Reset counter when hands are detected
                    else:
                        self.no_hands_counter += 1
                        if self.no_hands_counter >= self.max_no_hands_frames:
                            print("No hands detected for 1 second. Stopping recording...")
                            self.stop_recording()

                # Draw point history
                for i in range(1, len(self.recognizer.point_history)):
                    if i > 0:
                        cv.line(image, self.recognizer.point_history[i - 1],
                                self.recognizer.point_history[i], (0, 255, 0), 2)

                # Display recording status
                if self.recording:
                    cv.putText(image, f"Recording: {self.current_gesture}", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Display remaining frames before auto-stop
                    remaining_frames = self.max_no_hands_frames - self.no_hands_counter
                    cv.putText(image, f"Auto-stop in: {remaining_frames//30}s", (10, 70),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Display repetition progress
                    cv.putText(image, f"Repetition: {self.current_repetition + 1}/{self.target_repetitions}", (10, 110),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif self.waiting_for_next:
                    # Display countdown for next repetition
                    remaining_seconds = (self.wait_duration - self.wait_frames) // 30
                    cv.putText(image, f"Next repetition in: {remaining_seconds}s", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv.putText(image, f"Repetition: {self.current_repetition + 1}/{self.target_repetitions}", (10, 70),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display the frame
                cv.imshow('Data Collection', image)

                # Handle keyboard input
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and not self.recording and not self.waiting_for_next:
                    gesture_name = input("Enter gesture name: ")
                    if gesture_name:
                        try:
                            repetitions = int(input("Enter number of repetitions to collect (default: 1): ") or "1")
                            if repetitions < 1:
                                print("Number of repetitions must be at least 1. Using default value of 1.")
                                repetitions = 1
                            self.start_recording(gesture_name, repetitions)
                        except ValueError:
                            print("Invalid input. Using default value of 1 repetition.")
                            self.start_recording(gesture_name, 1)

        except Exception as e:
            print(f"Error during data collection: {e}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv.destroyAllWindows()
            print("Data collection session ended.")

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
            # Extract gesture name (remove repetition number and .json extension)
            gesture_name = '_'.join(filename.split('_')[:-1])
            
            # Add to gesture mappings
            if gesture_name not in gesture_mappings:
                gesture_mappings[gesture_name] = current_label
                current_label += 1
            
            # Load gesture data
            with open(os.path.join(data_dir, filename), 'r') as f:
                gesture_data = json.load(f)
                
            # Extract features and labels
            for frame_data in gesture_data:
                # Ensure features are a 1D array
                features = np.array(frame_data['features'])
                if len(features.shape) > 1:
                    features = features.flatten()
                X.append(features)
                y.append(gesture_mappings[gesture_name])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Print data statistics
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Number of classes: {len(gesture_mappings)}")
    print("Samples per class:")
    for gesture, label in gesture_mappings.items():
        count = np.sum(y == label)
        print(f"  {gesture}: {count} samples")
    
    # Save gesture mappings
    with open(os.path.join(data_dir, 'gesture_mappings.json'), 'w') as f:
        json.dump(gesture_mappings, f)
    
    return X, y, gesture_mappings

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
    model_path = os.path.join(save_dir, 'filipino_sign_language_model.keras')
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