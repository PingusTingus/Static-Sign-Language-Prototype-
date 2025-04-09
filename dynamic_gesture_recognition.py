import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib


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
        self.scaler = StandardScaler()

    def extract_dynamic_features(self, point_history):
        """Extract features from point history for dynamic gesture recognition"""
        if len(point_history) < self.feature_window:
            return None

        features = []
        points = np.array(point_history)

        # 1. Velocity features
        velocities = np.diff(points, axis=0)
        features.extend(np.mean(velocities, axis=0))
        features.extend(np.std(velocities, axis=0))

        # 2. Direction changes
        direction_changes = np.diff(np.sign(velocities), axis=0)
        features.extend(np.sum(np.abs(direction_changes), axis=0))

        # 3. Path curvature
        if len(points) > 2:
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            curvature = np.abs(dx * d2y - dy * d2x) / (dx * dx + dy * dy) ** 1.5
            features.extend([np.mean(curvature), np.max(curvature)])

        # 4. Spatial features
        features.extend([
            np.max(points[:, 0]) - np.min(points[:, 0]),  # x range
            np.max(points[:, 1]) - np.min(points[:, 1]),  # y range
            np.mean(np.sqrt(np.sum(velocities ** 2, axis=1)))  # average speed
        ])

        return np.array(features)

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
                        # Normalize features
                        features = self.scaler.fit_transform(features.reshape(1, -1))
                        gesture_info = {
                            'features': features,
                            'point': (x, y)
                        }

        return image, gesture_info

    def train_model(self, training_data, labels):
        """Train a model for dynamic gesture recognition"""
        # Prepare training data
        X = np.array([self.extract_dynamic_features(seq) for seq in training_data])
        y = np.array(labels)

        # Remove None values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        # Normalize features
        X = self.scaler.fit_transform(X)

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