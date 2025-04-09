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
        # Calculate expected feature dimension
        self.expected_features = self.feature_window * 2  # x,y coordinates
        self.expected_features += (self.feature_window - 1) * 2  # velocities
        self.expected_features += (self.feature_window - 2) * 2  # accelerations
        self.expected_features += (self.feature_window - 2)  # angles
        
        # Use MinMaxScaler instead of StandardScaler to avoid division issues
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_fitted = False

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

        try:
            # Convert point history to numpy array
            points = np.array(point_history[-self.feature_window:])  # Take only the last feature_window points

            # Calculate velocities and accelerations
            velocities = np.diff(points, axis=0)
            accelerations = np.diff(velocities, axis=0)

            # Calculate angles between consecutive points
            angles = []
            for i in range(len(points) - 2):
                v1 = points[i + 1] - points[i]
                v2 = points[i + 2] - points[i + 1]
                angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                angles.append(angle)

            # Combine features
            features = np.concatenate([
                points.flatten(),  # Position features (feature_window * 2)
                velocities.flatten(),  # Velocity features ((feature_window-1) * 2)
                accelerations.flatten(),  # Acceleration features ((feature_window-2) * 2)
                np.array(angles)  # Angle features (feature_window-2)
            ])

            # Ensure consistent feature dimension
            if len(features) != self.expected_features:
                # Pad or truncate to expected dimension
                if len(features) < self.expected_features:
                    features = np.pad(features, (0, self.expected_features - len(features)))
                else:
                    features = features[:self.expected_features]

            # Handle any NaN or inf values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

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
        
        # Ensure X is 2D
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Normalize features with safety check
        try:
            # Fit the scaler on the training data
            self.scaler.fit(X)
            self.scaler_fitted = True

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
        
        # Get number of unique classes
        num_classes = len(np.unique(y))
        
        # Create and train model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)),  # Use Input layer with correct shape
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Use a lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train with validation split and early stopping
        history = model.fit(
            X, y, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return model
    
    def save_model(self, model, path):
        """Save the trained model"""
        # Create models directory if it doesn't exist
        models_dir = os.path.join(self.base_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Add .keras extension to the path
        if not path.endswith('.keras'):
            path = path + '.keras'
            
        # Save model and scaler
        model.save(path)
        if self.scaler_fitted:
            scaler_path = path.replace('.keras', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        else:
            print("Warning: Scaler not fitted, skipping scaler save")
    
    def load_model(self, path):
        """Load a trained model"""
        # Add .keras extension if not present
        if not path.endswith('.keras'):
            path = path + '.keras'
            
        model = tf.keras.models.load_model(path)
        scaler_path = path.replace('.keras', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.scaler_fitted = True
        else:
            print("Warning: No scaler file found, using unfitted scaler")
            self.scaler_fitted = False
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