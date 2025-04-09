import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
import os

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class DynamicGestureRecognizer:
    def __init__(self, history_length=32, min_confidence=0.7):
        self.history_length = history_length
        self.min_confidence = min_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        
        # Initialize point history
        self.point_history = deque(maxlen=history_length)
        self.gesture_history = deque(maxlen=history_length)
        
        # Feature extraction parameters
        self.feature_window = 8  # Window size for feature extraction
        # Use MinMaxScaler instead of StandardScaler to avoid division issues
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Get base directory for model loading/saving
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
    def safe_divide(self, a, b, default=0.0):
        """Safely divide two arrays, returning default value for division by zero"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b, out=np.full_like(a, default, dtype=float), where=b!=0)
            return np.nan_to_num(result, nan=default, posinf=default, neginf=default)
    
    def extract_dynamic_features(self, point_history):
        """Extract features from point history for dynamic gesture recognition"""
        if len(point_history) < self.feature_window:
            return None
            
        features = []
        points = np.array(point_history)
        
        # 1. Velocity features (with safety checks)
        velocities = np.diff(points, axis=0)
        # Handle empty velocities array
        if len(velocities) == 0:
            velocities = np.zeros((1, 2))
        
        # Mean velocity with safety check
        mean_vel = np.mean(velocities, axis=0)
        mean_vel = np.nan_to_num(mean_vel, nan=0.0, posinf=0.0, neginf=0.0)
        features.extend(mean_vel)
        
        # Standard deviation with safety check
        std_vel = np.std(velocities, axis=0)
        std_vel = np.nan_to_num(std_vel, nan=0.0, posinf=0.0, neginf=0.0)
        features.extend(std_vel)
        
        # 2. Direction changes (with safety checks)
        if len(velocities) > 0:
            # Use a small epsilon to avoid division by zero
            epsilon = 1e-10
            # Calculate direction as unit vectors
            magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
            magnitudes = np.maximum(magnitudes, epsilon)  # Avoid division by zero
            directions = velocities / magnitudes[:, np.newaxis]
            
            # Calculate direction changes
            direction_changes = np.diff(np.sign(directions), axis=0)
            direction_changes = np.nan_to_num(direction_changes, nan=0.0, posinf=0.0, neginf=0.0)
            features.extend(np.sum(np.abs(direction_changes), axis=0))
        else:
            features.extend([0.0, 0.0])
        
        # 3. Path curvature (completely rewritten with safety checks)
        if len(points) > 2:
            # Calculate first derivatives
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            
            # Calculate second derivatives
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            # Calculate curvature using a safer formula
            # k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = np.abs(dx * d2y - dy * d2x)
            denominator = (dx * dx + dy * dy) ** 1.5
            
            # Replace zeros in denominator with a small number
            denominator = np.maximum(denominator, 1e-10)
            
            # Calculate curvature
            curvature = self.safe_divide(numerator, denominator)
            
            # Handle any remaining NaN or inf values
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Add curvature features
            features.extend([np.mean(curvature), np.max(curvature)])
        else:
            features.extend([0.0, 0.0])
        
        # 4. Spatial features (with safety checks)
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        # Calculate speed with safety check
        if len(velocities) > 0:
            speed = np.mean(np.sqrt(np.sum(velocities ** 2, axis=1)))
        else:
            speed = 0.0
        
        # Replace any NaN or inf values with 0
        x_range = np.nan_to_num(x_range, nan=0.0, posinf=0.0, neginf=0.0)
        y_range = np.nan_to_num(y_range, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
        
        features.extend([x_range, y_range, speed])
        
        # Convert to numpy array and handle any remaining NaN values
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def process_frame(self, frame):
        """Process a single frame and return gesture information"""
        # Convert to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image
        results = self.hands.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        gesture_info = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[8]
                x, y = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])
                
                # Add to point history
                self.point_history.append((x, y))
                
                # Extract features when we have enough points
                if len(self.point_history) >= self.feature_window:
                    features = self.extract_dynamic_features(list(self.point_history))
                    if features is not None:
                        # Use a try-except block for feature normalization
                        try:
                            # Reshape features for scaling
                            features_reshaped = features.reshape(1, -1)
                            
                            # Check if scaler is fitted
                            if not hasattr(self.scaler, 'n_features_in_'):
                                # First time scaling, fit the scaler
                                self.scaler.fit(features_reshaped)
                            
                            # Transform features
                            features_scaled = self.scaler.transform(features_reshaped)
                            
                            gesture_info = {
                                'features': features_scaled,
                                'point': (x, y)
                            }
                        except Exception as e:
                            print(f"Warning: Feature normalization failed: {e}")
                            # Use raw features if normalization fails
                            gesture_info = {
                                'features': features.reshape(1, -1),
                                'point': (x, y)
                            }
        
        return image, gesture_info
    
    def train_model(self, X, y):
        """Train a model for dynamic gesture recognition"""
        # Handle empty dataset
        if len(X) == 0:
            raise ValueError("No valid features extracted from training data")
        
        # Replace any NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features with safety check
        try:
            # Fit the scaler on the training data
            self.scaler.fit(X)
            # Transform the data
            X = self.scaler.transform(X)
        except Exception as e:
            print(f"Warning: Feature normalization failed: {e}")
            # Use a simple min-max scaling as fallback
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_range = X_max - X_min
            # Avoid division by zero
            X_range = np.maximum(X_range, 1e-10)
            X = (X - X_min) / X_range
        
        # Create and train model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
        
        return model
    
    def save_model(self, model, path):
        """Save the trained model"""
        # Create models directory if it doesn't exist
        models_dir = os.path.join(self.base_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Save model and scaler
        model.save(path)
        joblib.dump(self.scaler, path + '_scaler.pkl')
    
    def load_model(self, path):
        """Load a trained model"""
        model = tf.keras.models.load_model(path)
        self.scaler = joblib.load(path + '_scaler.pkl')
        return model

def main():
    # Initialize the recognizer
    recognizer = DynamicGestureRecognizer()
    
    # Initialize video capture
    cap = cv.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Process frame
        image, gesture_info = recognizer.process_frame(frame)
        
        # Draw point history
        for i in range(1, len(recognizer.point_history)):
            if i > 0:
                cv.line(image, recognizer.point_history[i - 1], recognizer.point_history[i],
                       (0, 255, 0), 2)
        
        # Display the frame
        cv.imshow('Dynamic Gesture Recognition', image)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main() 