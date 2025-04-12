import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Dict

class GestureModel:
    def __init__(self, input_shape: Tuple[int, int], num_classes: int):
        """
        Initialize the gesture recognition model.
        
        Args:
            input_shape: Shape of input features (sequence_length, feature_dim)
            num_classes: Number of gesture classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build the BiLSTM-GRU model."""
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )(inputs)
        x = layers.Dropout(0.3)(x)
        
        # GRU layer
        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        attention = layers.Dense(32, activation='relu')(x)
        attention = layers.Dense(1, activation='tanh')(attention)
        attention = layers.Flatten()(attention)
        attention_weights = layers.Activation('softmax')(attention)
        attention_weights = layers.RepeatVector(64)(attention_weights)
        attention_weights = layers.Permute([2, 1])(attention_weights)
        
        # Apply attention
        context = layers.Multiply()([x, attention_weights])
        context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(context)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              batch_size: int = 32, epochs: int = 50) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """Save the model to disk."""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load the model from disk."""
        self.model = tf.keras.models.load_model(filepath) 