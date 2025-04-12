import numpy as np
from typing import List, Tuple, Dict

class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor."""
        # Define hand landmark indices
        self.landmark_dict = {
            "WRIST": 0,
            "THUMB_CMC": 1,
            "THUMB_MCP": 2,
            "THUMB_IP": 3,
            "THUMB_TIP": 4,
            "INDEX_FINGER_MCP": 5,
            "INDEX_FINGER_PIP": 6,
            "INDEX_FINGER_DIP": 7,
            "INDEX_FINGER_TIP": 8,
            "MIDDLE_FINGER_MCP": 9,
            "MIDDLE_FINGER_PIP": 10,
            "MIDDLE_FINGER_DIP": 11,
            "MIDDLE_FINGER_TIP": 12,
            "RING_FINGER_MCP": 13,
            "RING_FINGER_PIP": 14,
            "RING_FINGER_DIP": 15,
            "RING_FINGER_TIP": 16,
            "PINKY_MCP": 17,
            "PINKY_PIP": 18,
            "PINKY_DIP": 19,
            "PINKY_TIP": 20
        }
        
    def extract_features(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from hand landmarks.
        
        Args:
            landmarks: Array of shape (sequence_length, 21, 3) containing hand landmarks
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract handshape features
        features['handshape'] = self._extract_handshape(landmarks)
        
        # Extract palm orientation
        features['orientation'] = self._extract_orientation(landmarks)
        
        # Extract location features
        features['location'] = self._extract_location(landmarks)
        
        # Extract movement features
        features['movement'] = self._extract_movement(landmarks)
        
        return features
    
    def _extract_handshape(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract handshape features."""
        # Calculate finger angles and distances
        thumb_angle = self._calculate_angle(
            landmarks[:, self.landmark_dict["THUMB_CMC"]],
            landmarks[:, self.landmark_dict["THUMB_MCP"]],
            landmarks[:, self.landmark_dict["THUMB_TIP"]]
        )
        
        index_angle = self._calculate_angle(
            landmarks[:, self.landmark_dict["INDEX_FINGER_MCP"]],
            landmarks[:, self.landmark_dict["INDEX_FINGER_PIP"]],
            landmarks[:, self.landmark_dict["INDEX_FINGER_TIP"]]
        )
        
        # Add more finger angles as needed
        
        return np.column_stack([thumb_angle, index_angle])
    
    def _extract_orientation(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract palm orientation features."""
        # Calculate palm normal vector
        palm_points = landmarks[:, [
            self.landmark_dict["WRIST"],
            self.landmark_dict["INDEX_FINGER_MCP"],
            self.landmark_dict["PINKY_MCP"]
        ]]
        
        v1 = palm_points[:, 1] - palm_points[:, 0]
        v2 = palm_points[:, 2] - palm_points[:, 0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
        
        return normal
    
    def _extract_location(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract location features."""
        # Use wrist position as reference point
        wrist_pos = landmarks[:, self.landmark_dict["WRIST"]]
        return wrist_pos
    
    def _extract_movement(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract movement features."""
        # Calculate velocity of wrist
        wrist_pos = landmarks[:, self.landmark_dict["WRIST"]]
        velocity = np.diff(wrist_pos, axis=0)
        # Pad the last frame to match sequence length
        velocity = np.pad(velocity, ((0, 1), (0, 0)), mode='edge')
        return velocity
    
    def _calculate_angle(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
        """Calculate angle between three points."""
        v1 = v1 - v2
        v3 = v3 - v2
        
        cos_angle = np.sum(v1 * v3, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v3, axis=1)
        )
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return angle 