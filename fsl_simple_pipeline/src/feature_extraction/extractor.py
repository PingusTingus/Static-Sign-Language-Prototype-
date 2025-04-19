import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import cv2

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor."""
        # Define feature dimensions
        self.handshape_dim = 21 * 3  # 21 landmarks * 3 coordinates
        self.movement_dim = 21 * 3  # 21 landmarks * 3 coordinates
        self.location_dim = 3  # x, y, z coordinates of hand center
        self.orientation_dim = 3  # pitch, roll, yaw angles
        
        # Total feature dimension
        self.feature_dim = self.handshape_dim + self.movement_dim + self.location_dim + self.orientation_dim
        
        # Initialize visualization parameters
        self.visualization_enabled = True
        self.feature_history = {
            'handshape': [],
            'movement': [],
            'location': [],
            'orientation': []
        }
        
        # Initialize state
        self.prev_landmarks = None

    def extract_handshape(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract handshape features from landmarks."""
        try:
            if landmarks is None or len(landmarks) == 0:
                return np.zeros(self.handshape_dim)
                
            # Normalize landmarks relative to wrist
            wrist = landmarks[0]
            normalized_landmarks = landmarks - wrist
            
            # Flatten and normalize
            handshape_features = normalized_landmarks.flatten()
            norm = np.linalg.norm(handshape_features)
            if norm > 0:
                handshape_features = handshape_features / norm
            
            return handshape_features
        except Exception as e:
            logger.error(f"Error extracting handshape features: {e}")
            return np.zeros(self.handshape_dim)

    def extract_movement(self, prev_landmarks: np.ndarray, current_landmarks: np.ndarray) -> np.ndarray:
        """Extract movement features between consecutive frames."""
        try:
            if prev_landmarks is None or current_landmarks is None:
                return np.zeros(self.movement_dim)
                
            # Calculate displacement
            displacement = current_landmarks - prev_landmarks
            
            # Calculate velocity
            velocity = displacement / (1/30)  # Assuming 30 FPS
            
            # Flatten and normalize
            movement_features = velocity.flatten()
            norm = np.linalg.norm(movement_features)
            if norm > 0:
                movement_features = movement_features / norm
            
            return movement_features
        except Exception as e:
            logger.error(f"Error extracting movement features: {e}")
            return np.zeros(self.movement_dim)

    def extract_location(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract location features (hand center position)."""
        try:
            if landmarks is None or len(landmarks) == 0:
                return np.zeros(self.location_dim)
                
            # Calculate hand center as mean of all landmarks
            hand_center = np.mean(landmarks, axis=0)
            
            # Normalize to [0, 1] range
            min_val = np.min(hand_center)
            max_val = np.max(hand_center)
            if max_val > min_val:
                location_features = (hand_center - min_val) / (max_val - min_val)
            else:
                location_features = np.zeros_like(hand_center)
            
            return location_features
        except Exception as e:
            logger.error(f"Error extracting location features: {e}")
            return np.zeros(self.location_dim)

    def extract_orientation(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract palm orientation features."""
        try:
            if landmarks is None or len(landmarks) < 21:
                return np.zeros(self.orientation_dim)
                
            # Calculate palm normal vector using thumb, index, and pinky
            thumb = landmarks[4]
            index = landmarks[8]
            pinky = landmarks[20]
            
            # Calculate two vectors in the palm plane
            v1 = index - thumb
            v2 = pinky - thumb
            
            # Calculate normal vector
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            
            # Convert to Euler angles
            pitch = np.arctan2(normal[1], normal[2])
            roll = np.arctan2(normal[0], normal[2])
            yaw = np.arctan2(normal[0], normal[1])
            
            return np.array([pitch, roll, yaw])
        except Exception as e:
            logger.error(f"Error extracting orientation features: {e}")
            return np.zeros(self.orientation_dim)

    def extract_features(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from hand landmarks.
        
        Args:
            landmarks: Array of hand landmarks
            
        Returns:
            Dictionary of extracted features
        """
        try:
            if landmarks is None or len(landmarks) == 0:
                logger.error("No landmarks provided")
                return {
                    'handshape': np.zeros(63),
                    'movement': np.zeros(63),
                    'location': np.zeros(3),
                    'orientation': np.zeros(3)
                }
            
            # Ensure landmarks are in the correct shape
            if len(landmarks.shape) == 2:
                landmarks = landmarks.reshape(1, *landmarks.shape)
            
            # Extract features for each frame
            features = {
                'handshape': [],
                'movement': [],
                'location': [],
                'orientation': []
            }
            
            prev_landmarks = None
            for frame_landmarks in landmarks:
                # Normalize landmarks
                normalized = self.normalize_landmarks(frame_landmarks)
                if normalized is None:
                    continue
                
                # Extract handshape features
                handshape = normalized.flatten()
                
                # Extract movement features
                movement = np.zeros(63)
                if prev_landmarks is not None:
                    prev_normalized = self.normalize_landmarks(prev_landmarks)
                    if prev_normalized is not None:
                        movement = (normalized - prev_normalized).flatten()
                
                # Extract location features
                location = np.mean(normalized, axis=0)
                
                # Extract orientation features
                thumb = normalized[4]
                index = normalized[8]
                pinky = normalized[20]
                v1 = index - thumb
                v2 = pinky - thumb
                orientation = np.cross(v1, v2)
                norm = np.linalg.norm(orientation)
                if norm > 0:
                    orientation = orientation / norm
                
                features['handshape'].append(handshape)
                features['movement'].append(movement)
                features['location'].append(location)
                features['orientation'].append(orientation)
                
                prev_landmarks = frame_landmarks
            
            # Convert lists to numpy arrays
            for key in features:
                features[key] = np.array(features[key])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {
                'handshape': np.zeros(63),
                'movement': np.zeros(63),
                'location': np.zeros(3),
                'orientation': np.zeros(3)
            }

    def visualize_features(self, frame: Optional[np.ndarray] = None):
        """Visualize extracted features on the frame."""
        if not self.visualization_enabled or frame is None:
            return
            
        try:
            y_offset = 100
            for feature_name, history in self.feature_history.items():
                if history:
                    # Calculate mean and std of recent features
                    recent_features = np.array(history[-10:])  # Last 10 frames
                    mean_features = np.mean(recent_features, axis=0)
                    std_features = np.std(recent_features, axis=0)
                    
                    # Display feature statistics
                    text = f"{feature_name}: mean={mean_features.mean():.3f}, std={std_features.mean():.3f}"
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 30
        except Exception as e:
            logger.error(f"Error visualizing features: {e}") 