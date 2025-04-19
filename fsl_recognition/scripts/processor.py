#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSL Gesture Sequence Recognition Model with Augmented Data Support
Trains and evaluates a model for recognizing complete gestures from sequences
Combines original and augmented data for more robust training

Created by: PingusTingus
Date: 2025-04-15 04:36:47
"""

import os
import sys
import json
import time
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RECORDED_GESTURES_DIR = os.path.join(DATA_DIR, 'recorded_gestures')
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, 'augmented_data')
FEATURES_DIR = os.path.join(DATA_DIR, 'extracted_features')
MODEL_SAVE_DIR = os.path.join(DATA_DIR, 'trained_models')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Print execution info
print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: PingusTingus")

# Options
MAX_SEQUENCE_LENGTH = 10  # Maximum number of states in a sequence
PADDING_VALUE = -1  # Value used for padding sequences to the same length


def load_all_gesture_data(use_augmentation=True):
    """
    Load all gesture recordings from the recorded_gestures directory
    and optionally include augmented data

    Args:
        use_augmentation: Whether to include augmented data in the dataset

    Returns:
        A list of gesture data and their labels
    """
    print("Loading gesture data...")

    # Initialize data containers
    all_gestures = []
    all_labels = []

    # Load original gesture recordings
    gesture_files = glob.glob(os.path.join(RECORDED_GESTURES_DIR, "*.json"))

    if not gesture_files:
        print("No original gesture recordings found in:", RECORDED_GESTURES_DIR)
    else:
        print(f"Found {len(gesture_files)} original gesture recordings")

        for file_path in tqdm(gesture_files, desc="Loading original gestures"):
            try:
                with open(file_path, 'r') as f:
                    gesture_data = json.load(f)

                # Extract the simplified sequence and gesture name
                if "simplified_sequence" in gesture_data and "name" in gesture_data:
                    sequence = gesture_data["simplified_sequence"]
                    name = gesture_data["name"]

                    # Only include sequences with at least one state
                    if sequence:
                        all_gestures.append(sequence)
                        all_labels.append(name)
                else:
                    print(f"Warning: Missing sequence or name in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Load augmented data if requested
    if use_augmentation:
        # First check if there are pre-extracted features available
        feature_files = glob.glob(os.path.join(FEATURES_DIR, "*.npz"))

        if feature_files:
            print(f"Found {len(feature_files)} pre-extracted feature files. These will be loaded separately.")

        # Load augmented gesture data files
        aug_files = glob.glob(os.path.join(AUGMENTED_DATA_DIR, "*.json"))

        if not aug_files:
            print("No augmented gesture recordings found in:", AUGMENTED_DATA_DIR)
        else:
            print(f"Found {len(aug_files)} augmented gesture files")

            for file_path in tqdm(aug_files, desc="Loading augmented gestures"):
                try:
                    with open(file_path, 'r') as f:
                        aug_data = json.load(f)

                    # Get the gesture name and sequences
                    original_gesture = aug_data.get("original_gesture")
                    sequences = aug_data.get("sequences", [])

                    if original_gesture and sequences:
                        # Add each augmented sequence to the dataset
                        for sequence in sequences:
                            if sequence:  # Make sure the sequence is valid
                                all_gestures.append(sequence)
                                all_labels.append(original_gesture)
                except Exception as e:
                    print(f"Error loading augmented data {file_path}: {e}")

    print(f"Successfully loaded {len(all_gestures)} total gesture sequences")
    print(f"Unique gesture classes: {len(set(all_labels))}")

    return all_gestures, all_labels


def load_preextracted_features():
    """
    Load pre-extracted features from FEATURES_DIR
    Returns the features, labels, and class names if available, or None if not found
    """
    feature_files = glob.glob(os.path.join(FEATURES_DIR, "combined_features_*.npz"))

    if not feature_files:
        feature_files = glob.glob(os.path.join(FEATURES_DIR, "*.npz"))

    if not feature_files:
        print("No pre-extracted feature files found.")
        return None, None, None

    # Use the most recent file (highest timestamp)
    latest_file = max(feature_files, key=os.path.getctime)
    print(f"Loading pre-extracted features from: {latest_file}")

    try:
        # Load features from NPZ file
        data = np.load(latest_file)
        X = data['X']
        y = data['y']
        classes = data['classes']

        print(f"Loaded pre-extracted features with shape {X.shape}")
        print(f"Found {len(np.unique(y))} unique classes")

        return X, y, classes
    except Exception as e:
        print(f"Error loading pre-extracted features: {e}")
        return None, None, None


def preprocess_gestures(gestures, labels):
    """
    Convert gesture sequences to numeric features and encode labels
    Returns preprocessed data ready for model training
    """
    print("Preprocessing gesture data...")

    # Create dictionaries for encoding handshapes, orientations, and movements
    handshape_set = set()
    orientation_set = set()
    movement_set = set()

    for gesture in gestures:
        for state in gesture:
            handshape_set.add(state.get('handshape', ''))
            orientation_set.add(state.get('orientation', ''))
            movement_set.add(state.get('movement', ''))

    # Convert sets to dictionaries with integer keys
    handshape_dict = {handshape: i for i, handshape in enumerate(sorted(handshape_set))}
    orientation_dict = {orientation: i for i, orientation in enumerate(sorted(orientation_set))}
    movement_dict = {movement: i for i, movement in enumerate(sorted(movement_set))}

    # Save encoding dictionaries
    encoding_data = {
        'handshape_dict': handshape_dict,
        'orientation_dict': orientation_dict,
        'movement_dict': movement_dict
    }

    with open(os.path.join(MODEL_SAVE_DIR, 'encoding_dicts.pkl'), 'wb') as f:
        pickle.dump(encoding_data, f)

    print(
        f"Encodings: {len(handshape_dict)} handshapes, {len(orientation_dict)} orientations, {len(movement_dict)} movements")

    # Convert sequences to feature matrices
    X = []
    for gesture in tqdm(gestures, desc="Extracting features"):
        sequence_features = []
        for state in gesture:
            handshape = handshape_dict.get(state.get('handshape', ''), 0)
            orientation = orientation_dict.get(state.get('orientation', ''), 0)
            movement = movement_dict.get(state.get('movement', ''), 0)
            confidence = state.get('confidence', 1.0)
            duration = state.get('duration', 0.0)

            # Combine features into a single vector for this state
            state_features = [handshape, orientation, movement, confidence, duration]
            sequence_features.append(state_features)

        # Pad or truncate sequence
        if len(sequence_features) > MAX_SEQUENCE_LENGTH:
            # Truncate to maximum length
            sequence_features = sequence_features[:MAX_SEQUENCE_LENGTH]
        elif len(sequence_features) < MAX_SEQUENCE_LENGTH:
            # Pad with padding value
            padding = [[PADDING_VALUE] * len(sequence_features[0]) for _ in
                       range(MAX_SEQUENCE_LENGTH - len(sequence_features))]
            sequence_features.extend(padding)

        X.append(sequence_features)

    # Convert to numpy array
    X = np.array(X)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Save label encoder
    with open(os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"Preprocessed {len(X)} sequences with {len(label_encoder.classes_)} unique gestures")

    # Also save as a NPZ file for future use
    timestamp = int(time.time())
    np.savez(os.path.join(FEATURES_DIR, f'preprocessed_features_{timestamp}.npz'),
             X=X, y=y, classes=label_encoder.classes_)

    return X, y, label_encoder.classes_


def build_lstm_model(input_shape, num_classes):
    """Build a LSTM-based model for sequence recognition"""
    print(f"Building LSTM model with input shape {input_shape} and {num_classes} output classes")

    model = keras.Sequential()

    # Masking layer to handle padding
    model.add(keras.layers.Masking(mask_value=PADDING_VALUE, input_shape=input_shape))

    # Bidirectional LSTM layers
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(32)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_transformer_model(input_shape, num_classes):
    """Build a Transformer-based model for sequence recognition"""
    print(f"Building transformer model with input shape {input_shape} and {num_classes} output classes")

    # Determine the feature dimension (input_shape[1])
    feature_dim = input_shape[1]

    # Input layer
    inputs = keras.layers.Input(shape=input_shape)

    # Masking for padding
    x = keras.layers.Masking(mask_value=PADDING_VALUE)(inputs)

    # Embedding layer to convert features to a higher-dimensional space
    embed_dim = 64
    x = keras.layers.Dense(embed_dim)(x)

    # Add positional encoding and self-attention
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=embed_dim // 4)(x, x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Feedforward network
    x = keras.layers.Dense(embed_dim * 2, activation='relu')(x)
    x = keras.layers.Dense(embed_dim)(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Global average pooling
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Final classification
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(X, y, model_type='lstm'):
    """
    Train a gesture recognition model
    Args:
        X: Preprocessed gesture data
        y: Encoded labels
        model_type: 'lstm' or 'transformer'
    """
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Build model based on type
    if model_type == 'transformer':
        model = build_transformer_model((MAX_SEQUENCE_LENGTH, X.shape[2]), len(np.unique(y)))
    else:
        model = build_lstm_model((MAX_SEQUENCE_LENGTH, X.shape[2]), len(np.unique(y)))

    # Model summary
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, f'gesture_model_{model_type}.h5'),
            save_best_only=True
        )
    ]

    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {evaluation[1]:.4f}")

    # Save model and metadata
    model.save(os.path.join(MODEL_SAVE_DIR, f'gesture_model_{model_type}_final.h5'))
    print(f"Model saved to {os.path.join(MODEL_SAVE_DIR, f'gesture_model_{model_type}_final.h5')}")

    # Plot training history
    plot_training_history(history, model_type)

    return model, X_test, y_test


def plot_training_history(history, model_type):
    """Plot training history curves"""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_type.upper()} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_type.upper()} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f'training_history_{model_type}.png'))
    print(f"Training history plot saved to {os.path.join(MODEL_SAVE_DIR, f'training_history_{model_type}.png')}")


def evaluate_model(model, X_test, y_test, label_names, model_type):
    """Evaluate model performance in more detail"""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_type.upper()} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f'confusion_matrix_{model_type}.png'))

    # Classification report
    print(f"\nClassification Report for {model_type.upper()} model:")
    report = classification_report(y_test, y_pred_classes, target_names=label_names)
    print(report)

    # Save report to file
    with open(os.path.join(MODEL_SAVE_DIR, f'classification_report_{model_type}.txt'), 'w') as f:
        f.write(f"Classification Report for {model_type.upper()} model:\n")
        f.write(report)

    # Class-wise accuracy
    class_accuracies = {}
    for i, label in enumerate(label_names):
        idx = np.where(y_test == i)[0]
        if len(idx) > 0:
            accuracy = np.mean(y_pred_classes[idx] == i)
            class_accuracies[label] = accuracy

    # Plot class accuracies
    plt.figure(figsize=(12, 6))
    classes = list(class_accuracies.keys())
    accuracies = [class_accuracies[c] for c in classes]

    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    classes = [classes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    plt.barh(classes, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title(f'{model_type.upper()} Class-wise Accuracy')
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f'class_accuracy_{model_type}.png'))

    return class_accuracies


def main():
    """Main function to train the gesture recognition model"""
    import argparse
    parser = argparse.ArgumentParser(description="Train FSL gesture recognition models with augmented data")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable use of augmented data")
    parser.add_argument("--only-preextracted", action="store_true", help="Use only pre-extracted features")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer", "both"], default="both",
                        help="Which model type to train (default: both)")
    args = parser.parse_args()

    # First check if we should use pre-extracted features
    if args.only_preextracted:
        print("Using only pre-extracted features...")
        X, y, label_names = load_preextracted_features()

        if X is None:
            print("No pre-extracted features found. Falling back to regular data loading.")
            args.only_preextracted = False

    # If not using pre-extracted features, load and preprocess data
    if not args.only_preextracted:
        # Load gesture data with augmentation based on args
        gestures, labels = load_all_gesture_data(use_augmentation=(not args.no_augmentation))

        if not gestures:
            print("No gesture data available. Please record gestures first.")
            return

        # Preprocess data
        X, y, label_names = preprocess_gestures(gestures, labels)

    # Train models based on selected option
    if args.model in ["lstm", "both"]:
        print("\nTraining LSTM model...")
        lstm_model, X_test_lstm, y_test_lstm = train_model(X, y, model_type='lstm')

        # Evaluate LSTM model
        print("\nEvaluating LSTM model:")
        lstm_accuracies = evaluate_model(lstm_model, X_test_lstm, y_test_lstm, label_names, "lstm")

    if args.model in ["transformer", "both"]:
        print("\nTraining Transformer model...")
        transformer_model, X_test_transformer, y_test_transformer = train_model(X, y, model_type='transformer')

        # Evaluate Transformer model
        print("\nEvaluating Transformer model:")
        transformer_accuracies = evaluate_model(transformer_model, X_test_transformer, y_test_transformer, label_names,
                                                "transformer")

    # Compare models if both were trained
    if args.model == "both":
        print("\nComparing Models:")
        common_gestures = set(lstm_accuracies.keys()) & set(transformer_accuracies.keys())

        better_lstm = []
        better_transformer = []

        for gesture in common_gestures:
            lstm_acc = lstm_accuracies[gesture]
            transformer_acc = transformer_accuracies[gesture]

            if lstm_acc > transformer_acc:
                better_lstm.append((gesture, lstm_acc, transformer_acc))
            elif transformer_acc > lstm_acc:
                better_transformer.append((gesture, lstm_acc, transformer_acc))

        if better_lstm:
            print("\nGestures where LSTM performed better:")
            for gesture, lstm_acc, transformer_acc in sorted(better_lstm, key=lambda x: x[1] - x[2], reverse=True):
                print(
                    f"  {gesture}: LSTM {lstm_acc:.4f} vs Transformer {transformer_acc:.4f} (diff: {lstm_acc - transformer_acc:.4f})")

        if better_transformer:
            print("\nGestures where Transformer performed better:")
            for gesture, lstm_acc, transformer_acc in sorted(better_transformer, key=lambda x: x[2] - x[1],
                                                             reverse=True):
                print(
                    f"  {gesture}: Transformer {transformer_acc:.4f} vs LSTM {lstm_acc:.4f} (diff: {transformer_acc - lstm_acc:.4f})")


if __name__ == "__main__":
    main()