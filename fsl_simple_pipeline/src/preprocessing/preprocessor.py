import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple

class Preprocessor:
    def __init__(self, window_size: int = 5, poly_order: int = 2):
        """
        Initialize the preprocessor.
        
        Args:
            window_size: Window size for Savitzky-Golay filter
            poly_order: Polynomial order for Savitzky-Golay filter
        """
        self.window_size = window_size
        self.poly_order = poly_order
        
    def preprocess(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Preprocess hand landmarks.
        
        Args:
            landmarks: Array of shape (sequence_length, 21, 3) containing hand landmarks
            
        Returns:
            Preprocessed landmarks
        """
        # Apply smoothing
        smoothed = self._smooth_sequence(landmarks)
        
        # Normalize coordinates
        normalized = self._normalize_coordinates(smoothed)
        
        return normalized
    
    def _smooth_sequence(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to the sequence."""
        smoothed = np.zeros_like(landmarks)
        
        # Apply smoothing to each coordinate
        for i in range(landmarks.shape[1]):  # For each landmark
            for j in range(landmarks.shape[2]):  # For each coordinate (x, y, z)
                smoothed[:, i, j] = savgol_filter(
                    landmarks[:, i, j],
                    window_length=self.window_size,
                    polyorder=self.poly_order
                )
        
        return smoothed
    
    def _normalize_coordinates(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize coordinates to [0, 1] range."""
        # Get min and max values for each coordinate
        min_vals = np.min(landmarks, axis=(0, 1), keepdims=True)
        max_vals = np.max(landmarks, axis=(0, 1), keepdims=True)
        
        # Normalize to [0, 1] range
        normalized = (landmarks - min_vals) / (max_vals - min_vals + 1e-8)
        
        return normalized 