#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSL Clean Sequence Recorder with Data Augmentation
Captures clean gesture sequences with smart filtering, editing, and data augmentation

Created by: PingusTingus
Date: 2025-04-15 03:55:12
"""

import os
import sys
import csv
import time
import copy
import json
import random
import itertools
import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
from collections import Counter, deque
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
KEYPOINT_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'keypoint_classifier', 'keypoint_classifier.tflite')
KEYPOINT_CLASSIFIER_LABEL = os.path.join(MODEL_DIR, 'keypoint_classifier', 'keypoint_classifier_label.csv')
RECORDED_GESTURES_DIR = os.path.join(DATA_DIR, 'recorded_gestures')
GESTURE_DEFINITIONS_PATH = os.path.join(DATA_DIR, 'gesture_definitions.json')
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, 'augmented_data')
FEATURES_DIR = os.path.join(DATA_DIR, 'extracted_features')

# Maximum sequence length used for padding/truncating
MAX_SEQUENCE_LENGTH = 10
PADDING_VALUE = -1

# Create directories if they don't exist
os.makedirs(os.path.join(MODEL_DIR, 'keypoint_classifier'), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RECORDED_GESTURES_DIR, exist_ok=True)
os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Print execution info
print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: PingusTingus")


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick
        self._difftimes.append(different_time)
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        return round(fps, 2)


class KeyPointClassifier(object):
    def __init__(self, model_path=KEYPOINT_CLASSIFIER_PATH, num_threads=1):
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found!")
            self.model_loaded = False
            return

        self.model_loaded = True
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Extract expected feature count
        input_shape = self.input_details[0]['shape']
        self.expected_feature_count = input_shape[1]
        print(f"Model expects {self.expected_feature_count} features")

    def __call__(self, landmark_list):
        if not self.model_loaded:
            return -1, 0.0, []

        # Ensure landmark_list has expected feature count
        if len(landmark_list) > self.expected_feature_count:
            landmark_list = landmark_list[:self.expected_feature_count]
        elif len(landmark_list) < self.expected_feature_count:
            # Pad with zeros if needed
            landmark_list = landmark_list + [0.0] * (self.expected_feature_count - len(landmark_list))

        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Get full result array for confidence values
        result_array = np.squeeze(result)
        result_index = np.argmax(result_array)
        confidence = float(result_array[result_index])

        return result_index, confidence, result_array


class DataAugmenter:
    """Class for augmenting gesture sequence data"""

    def __init__(self, augmentation_factor=5):
        """Initialize data augmenter with augmentation factor"""
        self.augmentation_factor = augmentation_factor

    def augment_sequence(self, sequence):
        """Generate augmented versions of a gesture sequence"""
        augmented_sequences = []

        # Always include the original sequence
        augmented_sequences.append(sequence)

        # Generate additional augmentations
        for i in range(self.augmentation_factor - 1):
            augmented_seq = self._create_augmentation(sequence)
            augmented_sequences.append(augmented_seq)

        return augmented_sequences

    def _create_augmentation(self, sequence):
        """Create a single augmented version of the sequence"""
        # Start with a copy of the original
        augmented_seq = copy.deepcopy(sequence)

        # Apply random transformations with varying probabilities
        if random.random() < 0.7:
            augmented_seq = self._time_stretch(augmented_seq)

        if random.random() < 0.5:
            augmented_seq = self._jitter_states(augmented_seq)

        if random.random() < 0.3:
            augmented_seq = self._drop_states(augmented_seq)

        if random.random() < 0.4:
            augmented_seq = self._modify_confidence(augmented_seq)

        # Ensure sequence integrity after transformations
        augmented_seq = self._ensure_sequence_integrity(augmented_seq)

        return augmented_seq

    def _time_stretch(self, sequence):
        """Stretch or compress the time duration of the sequence"""
        # Skip if too few states
        if len(sequence) <= 2:
            return sequence

        # Choose a random stretch factor between 0.8 and 1.2
        stretch_factor = random.uniform(0.8, 1.2)

        # Apply to all durations and timestamps
        current_time = 0
        for state in sequence:
            # Update start time
            state["start_time"] = current_time

            # Apply stretch factor to duration
            state["duration"] *= stretch_factor

            # Calculate new end time
            state["end_time"] = state["start_time"] + state["duration"]

            # Update current time for next state
            current_time = state["end_time"]

        return sequence

    def _jitter_states(self, sequence):
        """Add small variations to state properties"""
        for state in sequence:
            # Don't change the actual components (handshape, orientation, movement)
            # Just add small jitter to confidence and duration

            # Jitter confidence (within reasonable bounds)
            confidence_jitter = random.uniform(-0.1, 0.1)
            state["confidence"] = min(1.0, max(0.1, state["confidence"] + confidence_jitter))

            # Jitter duration slightly
            duration_jitter = random.uniform(-0.05, 0.05) * state["duration"]
            state["duration"] = max(0.1, state["duration"] + duration_jitter)

        return sequence

    def _drop_states(self, sequence):
        """Randomly drop a state (if sequence is long enough)"""
        # Only drop if we have more than 3 states to keep sequence meaningful
        if len(sequence) <= 3:
            return sequence

        # Select a non-first, non-last state to drop
        if len(sequence) > 2:
            drop_idx = random.randint(1, len(sequence) - 2)
            dropped_state = sequence.pop(drop_idx)

            # Ensure time continuity after dropping a state
            if drop_idx < len(sequence):
                # Update start time of the state after the dropped one
                sequence[drop_idx]["start_time"] = sequence[drop_idx - 1]["end_time"]
                sequence[drop_idx]["duration"] = sequence[drop_idx]["end_time"] - sequence[drop_idx]["start_time"]

        return sequence

    def _modify_confidence(self, sequence):
        """Modify confidence values in the sequence"""
        for state in sequence:
            # Apply a multiplier to confidence
            multiplier = random.uniform(0.9, 1.1)
            state["confidence"] = min(1.0, max(0.1, state["confidence"] * multiplier))

        return sequence

    def _ensure_sequence_integrity(self, sequence):
        """Ensure the sequence maintains proper ordering and time continuity"""
        if not sequence:
            return sequence

        # Ensure the sequence has proper timing
        current_time = 0
        for state in sequence:
            state["start_time"] = current_time
            state["duration"] = max(0.1, state["duration"])  # Ensure minimum duration
            state["end_time"] = state["start_time"] + state["duration"]
            current_time = state["end_time"]

        return sequence


class FeatureExtractor:
    """Extract features from gesture sequences for model training"""

    def __init__(self):
        """Initialize feature extractor"""
        self.handshape_set = set()
        self.orientation_set = set()
        self.movement_set = set()
        self.encoding_dicts = {}

    def build_feature_encodings(self, sequences):
        """Build encoding dictionaries from all gesture sequences"""
        print("Building feature encodings...")

        # Reset sets
        self.handshape_set = set()
        self.orientation_set = set()
        self.movement_set = set()

        # Collect all unique values
        for sequence in sequences:
            for state in sequence:
                self.handshape_set.add(state.get('handshape', ''))
                self.orientation_set.add(state.get('orientation', ''))
                self.movement_set.add(state.get('movement', ''))

        # Convert sets to dictionaries with integer keys
        self.encoding_dicts = {
            'handshape_dict': {handshape: i for i, handshape in enumerate(sorted(self.handshape_set))},
            'orientation_dict': {orientation: i for i, orientation in enumerate(sorted(self.orientation_set))},
            'movement_dict': {movement: i for i, movement in enumerate(sorted(self.movement_set))}
        }

        print(f"Encoding dictionaries built:")
        print(f"  • {len(self.encoding_dicts['handshape_dict'])} handshapes")
        print(f"  • {len(self.encoding_dicts['orientation_dict'])} orientations")
        print(f"  • {len(self.encoding_dicts['movement_dict'])} movements")

        # Save encoding dictionaries
        os.makedirs(os.path.join(FEATURES_DIR), exist_ok=True)
        with open(os.path.join(FEATURES_DIR, 'encoding_dicts.pkl'), 'wb') as f:
            import pickle
            pickle.dump(self.encoding_dicts, f)

        return self.encoding_dicts

    def extract_features(self, sequence):
        """Extract features from a single sequence"""
        # Ensure encoding dictionaries are built
        if not self.encoding_dicts:
            raise ValueError("Feature encodings not built. Call build_feature_encodings first.")

        handshape_dict = self.encoding_dicts['handshape_dict']
        orientation_dict = self.encoding_dicts['orientation_dict']
        movement_dict = self.encoding_dicts['movement_dict']

        # Create feature matrix
        sequence_features = []
        for state in sequence:
            # Get components
            handshape = state.get('handshape', '')
            orientation = state.get('orientation', '')
            movement = state.get('movement', '')
            confidence = state.get('confidence', 1.0)
            duration = state.get('duration', 0.0)

            # Convert to indices
            handshape_idx = handshape_dict.get(handshape, 0)
            orientation_idx = orientation_dict.get(orientation, 0)
            movement_idx = movement_dict.get(movement, 0)

            # Create feature vector
            state_features = [handshape_idx, orientation_idx, movement_idx, confidence, duration]
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

        return np.array(sequence_features)

    def extract_features_batch(self, sequences, labels):
        """Extract features from multiple sequences and their labels"""
        # Build encoding dictionaries if not already built
        if not self.encoding_dicts:
            self.build_feature_encodings(sequences)

        # Extract features from each sequence
        X = []
        for sequence in sequences:
            features = self.extract_features(sequence)
            X.append(features)

        # Convert to numpy array
        X = np.array(X)

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)

        # Save label encoder
        with open(os.path.join(FEATURES_DIR, 'label_encoder.pkl'), 'wb') as f:
            import pickle
            pickle.dump(label_encoder, f)

        print(f"Extracted features for {len(sequences)} sequences with {len(label_encoder.classes_)} unique labels")
        print(f"Feature shape: {X.shape}")

        return X, y, label_encoder.classes_


class GestureRecorder:
    """Records sequences of gesture components with smart filtering and augmentation"""

    def __init__(self):
        self.recording = False
        self.gesture_name = ""
        self.gesture_description = ""
        self.current_recording = []
        self.simplified_sequence = []
        self.start_time = 0
        self.recorded_gestures = self._load_recorded_gestures()
        self.stationary_start = None  # Track if the gesture started with stationary

        # Initialize augmenter and feature extractor
        self.augmenter = DataAugmenter(augmentation_factor=5)
        self.feature_extractor = FeatureExtractor()

    def _load_recorded_gestures(self):
        """Load existing gesture definitions"""
        if os.path.exists(GESTURE_DEFINITIONS_PATH):
            try:
                with open(GESTURE_DEFINITIONS_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading gesture definitions: {e}")
        return {}

    def _save_recorded_gestures(self):
        """Save gesture definitions to file"""
        with open(GESTURE_DEFINITIONS_PATH, 'w') as f:
            json.dump(self.recorded_gestures, f, indent=2)
        print(f"Saved gesture definitions to {GESTURE_DEFINITIONS_PATH}")

    def start_recording(self, gesture_name, description=""):
        """Start recording a new gesture sequence"""
        if self.recording:
            print("Already recording. Stop current recording first.")
            return False

        self.recording = True
        self.gesture_name = gesture_name
        self.gesture_description = description
        self.current_recording = []
        self.simplified_sequence = []
        self.start_time = time.time()
        self.stationary_start = None
        print(f"Started recording gesture: {gesture_name}")
        return True

    def add_frame(self, handshape, orientation, movement, confidence, handedness, timestamp=None):
        """Add a frame to the current recording"""
        if not self.recording:
            return False

        if timestamp is None:
            timestamp = time.time() - self.start_time

        frame_data = {
            "timestamp": timestamp,
            "handshape": handshape,
            "orientation": orientation,
            "movement": movement,
            "confidence": confidence,
            "handedness": handedness
        }

        self.current_recording.append(frame_data)

        # If this is the first frame, check if it's stationary
        if len(self.current_recording) == 1 and movement == "stationary":
            self.stationary_start = frame_data

        return True

    def finalize_recording(self):
        """Pre-process the recording before saving"""
        # First simplify the recording to get the sequence states
        self.simplified_sequence = self._simplify_recording()

        # Add stationary state at the end if needed
        self._ensure_stationary_ends()

        return self.simplified_sequence

    def stop_recording(self):
        """Stop recording and prepare for editing/saving"""
        if not self.recording:
            print("Not currently recording.")
            return None

        if not self.gesture_name:
            print("Cannot save recording without a name.")
            return None

        if not self.current_recording:
            print("No frames recorded.")
            self.recording = False
            return None

        self.recording = False

        # Finalize the recording
        self.finalized_sequence = self.finalize_recording()

        print(f"Recording stopped. {len(self.finalized_sequence)} sequence states detected.")
        print("You can now review and edit the sequence before saving.")

        return self.finalized_sequence

    def save_recording(self, perform_augmentation=True):
        """Save the edited sequence with optional augmentation"""
        if self.recording:
            print("Cannot save while still recording.")
            return None

        if not self.gesture_name:
            print("Cannot save recording without a name.")
            return None

        if not hasattr(self, 'finalized_sequence') or not self.finalized_sequence:
            print("No finalized sequence to save.")
            return None

        # Create gesture data
        gesture_data = {
            "name": self.gesture_name,
            "description": self.gesture_description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": self.finalized_sequence[-1]["end_time"] if self.finalized_sequence else 0,
            "frame_count": len(self.current_recording),
            "components": self.current_recording,
            "simplified_sequence": self.finalized_sequence
        }

        # Save to recordings directory
        filename = f"{self.gesture_name.lower().replace(' ', '_')}_{int(time.time())}.json"
        file_path = os.path.join(RECORDED_GESTURES_DIR, filename)

        with open(file_path, 'w') as f:
            json.dump(gesture_data, f, indent=2)

        print(f"Saved recording to {file_path}")

        # Add to gesture definitions
        self.recorded_gestures[self.gesture_name] = {
            "description": self.gesture_description,
            "simplified_sequence": self.finalized_sequence,
            "file": filename
        }

        self._save_recorded_gestures()

        # Generate augmented data if requested
        if perform_augmentation:
            self.generate_augmentations(self.gesture_name, self.finalized_sequence)

        # Reset recording state
        recording_copy = copy.deepcopy(self.current_recording)
        simplified_copy = copy.deepcopy(self.finalized_sequence)
        self.current_recording = []
        self.simplified_sequence = []
        if hasattr(self, 'finalized_sequence'):
            delattr(self, 'finalized_sequence')

        return simplified_copy

    def generate_augmentations(self, gesture_name, sequence):
        """Generate augmented versions of the sequence and save them"""
        print(f"Generating augmented sequences for '{gesture_name}'...")

        # Generate augmentations
        augmented_sequences = self.augmenter.augment_sequence(sequence)

        # Create augmentation data
        aug_data = {
            "original_gesture": gesture_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "augmentation_count": len(augmented_sequences),
            "sequences": augmented_sequences
        }

        # Save augmentations
        aug_filename = f"{gesture_name.lower().replace(' ', '_')}_augmented_{int(time.time())}.json"
        aug_file_path = os.path.join(AUGMENTED_DATA_DIR, aug_filename)

        with open(aug_file_path, 'w') as f:
            json.dump(aug_data, f, indent=2)

        print(f"Saved {len(augmented_sequences)} augmented sequences to {aug_file_path}")

        # Extract features from augmented sequences
        self.extract_features_from_augmentations(gesture_name, augmented_sequences)

        return len(augmented_sequences)

    def extract_features_from_augmentations(self, gesture_name, sequences):
        """Extract features from augmented sequences for model training"""
        print(f"Extracting features from augmented sequences for '{gesture_name}'...")

        # Create labels (all from same gesture class)
        labels = [gesture_name] * len(sequences)

        # Extract features
        try:
            X, y, classes = self.feature_extractor.extract_features_batch(sequences, labels)

            # Save features
            features_filename = f"{gesture_name.lower().replace(' ', '_')}_features_{int(time.time())}.npz"
            features_file_path = os.path.join(FEATURES_DIR, features_filename)

            np.savez(features_file_path, X=X, y=y, classes=classes)

            print(f"Saved extracted features to {features_file_path}")
            return True

        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _simplify_recording(self, min_duration=0.2):
        """Simplify the recording to key components with significant durations,
        ensuring no consecutive states are identical"""
        if not self.current_recording:
            return []

        simplified = []
        current_state = None
        state_start = 0

        for i, frame in enumerate(self.current_recording):
            # Create state identifier tuple
            state = (frame["handshape"], frame["orientation"], frame["movement"])
            timestamp = frame["timestamp"]
            handedness = frame["handedness"]
            confidence = frame["confidence"]

            # First frame
            if current_state is None:
                current_state = state
                state_data = {
                    "handshape": frame["handshape"],
                    "orientation": frame["orientation"],
                    "movement": frame["movement"],
                    "handedness": handedness,
                    "confidence": confidence,
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "duration": 0
                }
                state_start = timestamp
                continue

            # State changed
            if state != current_state:
                # Only include if lasted long enough
                duration = timestamp - state_start
                if duration >= min_duration:
                    # Update the end time and duration
                    state_data["end_time"] = timestamp
                    state_data["duration"] = duration

                    # Add to simplified sequence if it's different from previous state
                    if not simplified or simplified[-1]["handshape"] != state_data["handshape"] or \
                            simplified[-1]["orientation"] != state_data["orientation"] or \
                            simplified[-1]["movement"] != state_data["movement"]:
                        simplified.append(state_data)

                # Start a new state
                current_state = state
                state_data = {
                    "handshape": frame["handshape"],
                    "orientation": frame["orientation"],
                    "movement": frame["movement"],
                    "handedness": handedness,
                    "confidence": confidence,
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "duration": 0
                }
                state_start = timestamp

        # Add final state
        if current_state and len(self.current_recording) > 0:
            end_time = self.current_recording[-1]["timestamp"]
            duration = end_time - state_start
            if duration >= min_duration:
                # Update the end time and duration
                state_data["end_time"] = end_time
                state_data["duration"] = duration

                # Add to simplified sequence if it's different from previous state
                if not simplified or simplified[-1]["handshape"] != state_data["handshape"] or \
                        simplified[-1]["orientation"] != state_data["orientation"] or \
                        simplified[-1]["movement"] != state_data["movement"]:
                    simplified.append(state_data)

        # Filter out very short sequences (likely noise)
        simplified = [s for s in simplified if s["duration"] >= min_duration]

        return simplified

    def _ensure_stationary_ends(self):
        """Ensure the sequence starts and ends with stationary components if possible"""
        if not self.simplified_sequence:
            return

        # If we recorded a stationary start but it's not in simplified sequence, add it
        if self.stationary_start and (not self.simplified_sequence or
                                      self.simplified_sequence[0]["movement"] != "stationary"):
            # Create a short stationary state at the start
            start_state = {
                "handshape": self.stationary_start["handshape"],
                "orientation": self.stationary_start["orientation"],
                "movement": "stationary",
                "handedness": self.stationary_start["handedness"],
                "confidence": self.stationary_start["confidence"],
                "start_time": 0,
                "end_time": 0.2,
                "duration": 0.2
            }
            self.simplified_sequence.insert(0, start_state)

            # Adjust the next state's start time
            if len(self.simplified_sequence) > 1:
                self.simplified_sequence[1]["start_time"] = start_state["end_time"]
                self.simplified_sequence[1]["duration"] = (self.simplified_sequence[1]["end_time"] -
                                                           self.simplified_sequence[1]["start_time"])

        # If the last state isn't stationary, add one with the same shape and orientation
        if self.simplified_sequence and self.simplified_sequence[-1]["movement"] != "stationary":
            last_state = self.simplified_sequence[-1]
            end_state = {
                "handshape": last_state["handshape"],
                "orientation": last_state["orientation"],
                "movement": "stationary",
                "handedness": last_state["handedness"],
                "confidence": last_state["confidence"],
                "start_time": last_state["end_time"],
                "end_time": last_state["end_time"] + 0.2,
                "duration": 0.2
            }
            self.simplified_sequence.append(end_state)

    def edit_sequence(self):
        """Edit the simplified sequence before saving"""
        # We'll implement a simple terminal-based editor here
        if not hasattr(self, 'finalized_sequence') or not self.finalized_sequence:
            print("No sequence to edit.")
            return False

        sequence = self.finalized_sequence

        print("\n=== Sequence Editor ===")
        print(f"Gesture: {self.gesture_name}")
        print(f"Total states: {len(sequence)}\n")

        # Display the sequence
        self._display_sequence(sequence)

        while True:
            print("\nOptions:")
            print("  [d #] - Delete state # (e.g., 'd 3' to delete state 3)")
            print("  [m # #] - Move state from position # to position # (e.g., 'm 2 4')")
            print("  [v] - View current sequence")
            print("  [s] - Save changes and exit")
            print("  [c] - Cancel editing without saving")

            command = input("\nEnter command: ").strip().lower()

            if command.startswith('d '):
                try:
                    idx = int(command.split()[1]) - 1  # Convert to 0-based index
                    if 0 <= idx < len(sequence):
                        removed = sequence.pop(idx)
                        print(
                            f"Removed state {idx + 1}: [{removed['handshape']}, {removed['orientation']}, {removed['movement']}]")
                        # Adjust timestamps
                        self._recalculate_timestamps(sequence)
                    else:
                        print(f"Invalid state number. Must be between 1 and {len(sequence)}")
                except (ValueError, IndexError):
                    print("Invalid delete command. Use format: d #")

            elif command.startswith('m '):
                try:
                    parts = command.split()
                    from_idx = int(parts[1]) - 1  # Convert to 0-based index
                    to_idx = int(parts[2]) - 1  # Convert to 0-based index

                    if 0 <= from_idx < len(sequence) and 0 <= to_idx < len(sequence):
                        state = sequence.pop(from_idx)
                        sequence.insert(to_idx, state)
                        print(f"Moved state from position {from_idx + 1} to {to_idx + 1}")
                        # Adjust timestamps
                        self._recalculate_timestamps(sequence)
                    else:
                        print(f"Invalid position. Must be between 1 and {len(sequence)}")
                except (ValueError, IndexError):
                    print("Invalid move command. Use format: m # #")

            elif command == 'v':
                self._display_sequence(sequence)

            elif command == 's':
                self.finalized_sequence = sequence
                print("Changes saved.")
                return True

            elif command == 'c':
                print("Editing cancelled.")
                return False

            else:
                print("Unknown command.")

    def _display_sequence(self, sequence):
        """Display the sequence in a readable format"""
        print("\nCurrent sequence:")
        print("--------------------------------------------------")
        print("# | Handshape | Orientation | Movement | Duration")
        print("--------------------------------------------------")

        for i, state in enumerate(sequence):
            print(
                f"{i + 1:2} | {state['handshape']:9} | {state['orientation']:10} | {state['movement']:8} | {state['duration']:.2f}s")

        print("--------------------------------------------------")

    def _recalculate_timestamps(self, sequence):
        """Recalculate timestamps after editing the sequence"""
        if not sequence:
            return

        # Start at time 0
        current_time = 0

        for i, state in enumerate(sequence):
            # Set start time to current time
            state["start_time"] = current_time

            # Keep the original duration
            # But ensure minimum duration
            state["duration"] = max(state["duration"], 0.2)

            # Calculate end time
            state["end_time"] = state["start_time"] + state["duration"]

            # Update current time for next state
            current_time = state["end_time"]

    def is_recording(self):
        """Check if currently recording"""
        return self.recording

    def get_recorded_duration(self):
        """Get current recording duration in seconds"""
        if not self.recording:
            return 0
        return time.time() - self.start_time

    def get_frame_count(self):
        """Get number of frames in current recording"""
        return len(self.current_recording)

    def cancel_recording(self):
        """Cancel current recording without saving"""
        if not self.recording:
            return False

        self.recording = False
        self.current_recording = []
        self.simplified_sequence = []
        if hasattr(self, 'finalized_sequence'):
            delattr(self, 'finalized_sequence')
        print("Recording canceled.")
        return True

    def process_all_for_training(self):
        """Process all recorded gestures for training, with augmentation and feature extraction"""
        print("\nProcessing all recorded gestures for training...")

        # Step 1: Gather all recorded gestures
        all_gestures = []
        all_sequences = []
        all_labels = []

        for gesture_name, info in self.recorded_gestures.items():
            file_path = os.path.join(RECORDED_GESTURES_DIR, info["file"])
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        gesture_data = json.load(f)

                    sequence = gesture_data.get("simplified_sequence", [])
                    if sequence:
                        all_gestures.append(gesture_name)
                        all_sequences.append(sequence)
                        all_labels.append(gesture_name)
                except Exception as e:
                    print(f"Error loading gesture file {file_path}: {e}")

        if not all_sequences:
            print("No recorded gestures found for processing.")
            return False

        print(f"Found {len(all_sequences)} recorded gestures")

        # Step 2: Generate augmentations for all gestures
        augmented_sequences = []
        augmented_labels = []

        print("Generating augmentations...")
        for i, (gesture, sequence) in enumerate(zip(all_gestures, all_sequences)):
            print(f"Processing gesture {i + 1}/{len(all_gestures)}: {gesture}")

            # Generate augmentations for this sequence
            augmented = self.augmenter.augment_sequence(sequence)

            # Add to collection
            augmented_sequences.extend(augmented)
            augmented_labels.extend([gesture] * len(augmented))

        print(f"Generated {len(augmented_sequences)} augmented sequences total")

        # Step 3: Extract features from all augmented sequences
        print("\nExtracting features from all sequences...")
        try:
            # Build feature encodings from all sequences
            self.feature_extractor.build_feature_encodings(augmented_sequences)

            # Extract features
            X, y, classes = self.feature_extractor.extract_features_batch(augmented_sequences, augmented_labels)

            # Save combined features
            features_filename = f"combined_features_{int(time.time())}.npz"
            features_file_path = os.path.join(FEATURES_DIR, features_filename)

            np.savez(features_file_path, X=X, y=y, classes=classes)

            print(f"\nSaved combined features for all gestures to {features_file_path}")
            print(f"Feature matrix shape: {X.shape}")
            print(f"Number of classes: {len(classes)}")
            print(f"Classes: {', '.join(classes)}")

            return True

        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return False


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--model_selection", type=int, default=1)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    parser.add_argument("--max_hands", type=int, default=2)
    parser.add_argument("--augmentation_factor", type=int, default=5,
                        help="Number of augmented versions to generate per recorded gesture")

    return parser.parse_args()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0.0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def improve_prediction(hand_sign_id, confidence, result_array, landmark_list, labels):
    """Better prediction logic focused on O/C differentiation"""
    # Only improve O/C recognition - for simplicity and performance
    if len(landmark_list) < 21:
        return hand_sign_id, confidence

    # Find O and C indices in labels
    o_index, c_index = -1, -1
    for i, label in enumerate(labels):
        if label.lower() == 'o':
            o_index = i
        elif label.lower() == 'c':
            c_index = i

    # If we can't find both O and C in the labels, return original prediction
    if o_index == -1 or c_index == -1:
        return hand_sign_id, confidence

    # Only check O/C confusion if one of them is predicted or both have high confidence
    current_label = labels[hand_sign_id].lower() if 0 <= hand_sign_id < len(labels) else ""
    if current_label not in ['o', 'c']:
        # Check if O and C are both possible alternatives
        o_confidence = result_array[o_index] if o_index < len(result_array) else 0
        c_confidence = result_array[c_index] if c_index < len(result_array) else 0
        # If neither is a strong alternative, keep original prediction
        if max(o_confidence, c_confidence) < 0.3:
            return hand_sign_id, confidence

    # Simple check for O vs C: measure thumb-index distance
    thumb_tip = np.array(landmark_list[4][:2])  # thumb tip (x,y)
    index_tip = np.array(landmark_list[8][:2])  # index finger tip
    wrist = np.array(landmark_list[0][:2])
    middle_mcp = np.array(landmark_list[9][:2])

    # Get hand size for normalization
    hand_size = np.linalg.norm(middle_mcp - wrist)
    if hand_size == 0:
        hand_size = 1

    # Normalized thumb-index distance - key for O vs C
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip) / hand_size

    # Clear cases: if distance is very small -> O, if larger -> C
    if thumb_index_dist < 0.1:  # Clearly O (touching)
        return o_index, 0.9
    elif thumb_index_dist > 0.25:  # Clearly C (gap)
        return c_index, 0.9
    elif current_label in ['o', 'c']:
        # For borderline cases, look at the raw model prediction
        if current_label == 'o' and thumb_index_dist < 0.15:
            # Model says O and distance is small - confirm O
            return o_index, confidence
        elif current_label == 'c' and thumb_index_dist > 0.15:
            # Model says C and distance is larger - confirm C
            return c_index, confidence

    # Fall back to the model's prediction
    return hand_sign_id, confidence


def detect_orientation(landmark_list, handedness="Right"):
    """Simple orientation detection"""
    if len(landmark_list) < 21:
        return "unknown"

    # Use reference points that are stable
    wrist = np.array(landmark_list[0])
    index_mcp = np.array(landmark_list[5])
    pinky_mcp = np.array(landmark_list[17])

    # Create a plane using these stable points
    vector1 = index_mcp - wrist
    vector2 = pinky_mcp - wrist

    # Calculate normal vector to the palm plane
    normal = np.cross(vector1, vector2)
    normal_magnitude = np.linalg.norm(normal)

    if normal_magnitude > 0:
        normal = normal / normal_magnitude
    else:
        # Fall back to simplistic estimation if normal calculation fails
        # For example, if fingers are too close together
        if handedness == "Right":
            normal = np.array([-1, 0, 0])
        else:
            normal = np.array([1, 0, 0])

    # Extract components
    nx, ny, nz = normal

    # Find the major axis
    axis = np.argmax(np.abs([nx, ny, nz]))

    # For z-axis (front/back)
    if axis == 2:
        # Primarily facing toward/away from camera
        return "back" if nz > 0 else "front"

    # For y-axis (up/down)
    elif axis == 1:
        # Primarily facing up/down
        return "up" if ny < 0 else "down"

    # For x-axis (left/right)
    else:
        # Normalize based on handedness
        is_right_hand = handedness == "Right"
        if (nx > 0 and is_right_hand) or (nx < 0 and not is_right_hand):
            return "right"
        else:
            return "left"


def detect_simplified_movement(position_history, min_movement=5.0):
    """
    Detect simplified hand movement directions
    Returns one of: stationary, up, down, left, right, forward, backward
    """
    if len(position_history) < 5:
        return "stationary"

    # Get recent positions using list() conversion to handle deque safely
    positions = list(position_history)
    if len(positions) < 5:
        return "stationary"

    # Get recent positions (average a few frames to reduce noise)
    recent_positions = positions[-3:]
    past_positions = positions[-8:-5] if len(positions) >= 8 else positions[:3]

    # Convert to numpy arrays and compute means
    try:
        recent_pos = np.mean([np.array(p) for p in recent_positions], axis=0)
        past_pos = np.mean([np.array(p) for p in past_positions], axis=0)
    except:
        return "stationary"  # If error in calculation

    # Calculate movement vector
    movement = recent_pos - past_pos

    # Check if movement is large enough
    distance = np.linalg.norm(movement)
    if distance < min_movement:
        return "stationary"

    # Determine dominant direction
    x_move, y_move, z_move = movement

    # Determine primary movement axis
    if abs(x_move) > max(abs(y_move), abs(z_move)):
        # Left-right movement is dominant
        return "right" if x_move > 0 else "left"
    elif abs(y_move) > abs(z_move):
        # Up-down movement is dominant
        return "up" if y_move < 0 else "down"  # y is inverted in image coordinates
    else:
        # Forward-backward movement is dominant
        return "forward" if z_move < 0 else "backward"


def draw_landmarks(image, landmark_list, handedness="Right"):
    """Draw hand landmarks with handedness indicator"""
    # Choose color based on handedness
    color = (0, 255, 0) if handedness == "Right" else (255, 0, 255)  # Green for right, magenta for left

    # Draw the hand skeleton
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (0, 5), (5, 9), (9, 13), (13, 17)  # palm
    ]

    # Draw connections
    for connection in connections:
        if len(landmark_list) > connection[1]:
            try:
                point1 = (int(landmark_list[connection[0]][0]), int(landmark_list[connection[0]][1]))
                point2 = (int(landmark_list[connection[1]][0]), int(landmark_list[connection[1]][1]))
                cv.line(image, point1, point2, color, 2)
            except:
                continue

    # Draw landmark points
    for idx, landmark in enumerate(landmark_list):
        if idx < len(landmark_list):
            try:
                pos = (int(landmark[0]), int(landmark[1]))
                if idx in [4, 8, 12, 16, 20]:  # Fingertips
                    cv.circle(image, pos, 7, (255, 0, 0), -1)
                else:
                    cv.circle(image, pos, 5, color, -1)
            except:
                continue

    # Add handedness label near the wrist
    if len(landmark_list) > 0:
        try:
            wrist_pos = (int(landmark_list[0][0]), int(landmark_list[0][1]))
            cv.putText(image, handedness,
                       (wrist_pos[0] - 30, wrist_pos[1] + 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
        except:
            pass

    return image


def draw_info_text(image, handshape_text, orientation_text, movement_text, brect, confidence, handedness):
    """Draw information about hand sign, orientation and movement"""
    # Choose color based on handedness
    color = (0, 255, 0) if handedness == "Right" else (255, 0, 255)  # Green for right, magenta for left

    # Draw background box
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 85),
                 (40, 40, 40), -1)

    # Draw handedness indicator
    hand_text = f"{handedness} Hand"
    cv.putText(image, hand_text, (brect[0] + 5, brect[1] - 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv.LINE_AA)

    # Draw hand sign
    cv.putText(image, f"Sign: {handshape_text} ({confidence:.2f})", (brect[0] + 5, brect[1] - 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Draw orientation
    cv.putText(image, f"Orientation: {orientation_text}", (brect[0] + 5, brect[1] - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Draw movement
    cv.putText(image, f"Movement: {movement_text}", (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_recording_status(image, recorder, y_position=70):
    """Draw the recording status with sequence count"""
    if not recorder.is_recording():
        # Not recording - show instructions
        cv.putText(image, "Press 'r' to start recording", (10, y_position),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        # If we have a finalized sequence waiting for edit/save
        if hasattr(recorder, 'finalized_sequence') and recorder.finalized_sequence:
            cv.putText(image, "Press 'e' to edit sequence, 's' to save", (10, y_position + 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv.LINE_AA)

        return image

    # Calculate progress bar dimensions
    bar_width = 200
    bar_height = 20
    bar_x = 10
    bar_y = y_position + 10

    # Draw background
    cv.rectangle(image, (bar_x - 5, bar_y - 25), (bar_x + bar_width + 5, bar_y + bar_height + 25),
                 (80, 0, 0), -1)

    # Draw recording indicator
    cv.circle(image, (bar_x + 10, bar_y - 10), 8, (0, 0, 255), -1)

    # Draw recording info
    cv.putText(image, f"Recording: '{recorder.gesture_name}'", (bar_x + 25, bar_y - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

    # Draw progress bar
    duration = recorder.get_recorded_duration()
    frames = recorder.get_frame_count()
    cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

    # Fill based on time (assume 10 seconds max)
    fill_width = min(int(duration / 10.0 * bar_width), bar_width)
    cv.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 0, 255), -1)

    # Show time and frames
    cv.putText(image, f"Time: {duration:.1f}s | Frames: {frames}", (bar_x, bar_y + bar_height + 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Show instructions
    cv.putText(image, "Press 's' to stop and edit", (bar_x + bar_width + 15, bar_y + 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_component_history(image, component_history, max_display=10, start_y=200):
    """Visualize the component history"""
    if not component_history:
        return image

    # Background
    cv.rectangle(image, (10, start_y - 10), (480, start_y + 30 * min(max_display, len(component_history))),
                 (60, 60, 60), -1)

    # Title
    cv.putText(image, "Component History:", (15, start_y - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)

    # Show most recent items first (reverse)
    display_items = list(component_history)[-max_display:]

    for i, component in enumerate(display_items):
        y_pos = start_y + 20 * i

        # Create formatted component text
        handshape = component.get("handshape", "?")
        orientation = component.get("orientation", "?")
        movement = component.get("movement", "?")
        handedness = component.get("handedness", "?")

        # Choose color based on handedness
        text_color = (150, 255, 150) if handedness == "Right" else (255, 150, 255)

        component_text = f"{handedness[0]}: [{handshape}, {orientation}, {movement}]"
        cv.putText(image, component_text, (20, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)

    return image


def draw_sequence_visualization(image, sequence, start_y=350):
    """Draw the final sequence visualization"""
    if not sequence:
        return image

    # Background - calculate height based on sequence length
    height = min(30 * len(sequence) + 50, 200)
    cv.rectangle(image, (10, start_y), (480, start_y + height), (40, 40, 60), -1)

    # Title
    cv.putText(image, "Final Gesture Sequence:", (20, start_y + 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1, cv.LINE_AA)

    # Draw each state in the sequence
    for i, state in enumerate(sequence):
        y_pos = start_y + 50 + i * 25

        # Format components
        handshape = state.get("handshape", "?")
        orientation = state.get("orientation", "?")
        movement = state.get("movement", "?")
        duration = state.get("duration", 0)

        # Generate a color based on movement
        if movement == "stationary":
            color = (150, 150, 255)  # Stationary - blue
        else:
            color = (150, 255, 150)  # Moving - green

        # Draw state
        state_text = f"{i + 1}: [{handshape}, {orientation}, {movement}] {duration:.1f}s"
        cv.putText(image, state_text, (20, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

        # Stop if we reach the bottom of the area
        if y_pos > start_y + height - 25:
            cv.putText(image, f"...{len(sequence) - i - 1} more states...", (20, y_pos + 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)
            break

    return image


def draw_augmentation_info(image, start_y=550):
    """Draw information about augmentation and feature extraction"""
    # Background
    cv.rectangle(image, (10, start_y), (480, start_y + 120), (40, 40, 80), -1)

    # Title
    cv.putText(image, "Data Augmentation & Feature Extraction:", (20, start_y + 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1, cv.LINE_AA)

    # Info
    info_lines = [
        "• On save: 5 augmented versions created per gesture",
        "• Variations include: time stretch, jitter, state removal",
        "• Features extracted automatically for model training",
        "• Press 'p' to process all gestures for combined training"
    ]

    for i, line in enumerate(info_lines):
        y_pos = start_y + 50 + i * 20
        cv.putText(image, line, (20, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv.LINE_AA)

    return image


def show_help_window():
    """Show help window with instructions"""
    help_img = np.zeros((700, 800, 3), dtype=np.uint8)

    # Title
    cv.putText(help_img, "FSL Clean Sequence Recorder with Data Augmentation", (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    # Controls section
    cv.putText(help_img, "Controls:", (20, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1, cv.LINE_AA)

    controls = [
        "'r' - Start recording a new gesture",
        "'s' - Stop recording and prepare for editing",
        "'e' - Edit the recorded sequence",
        "'a' - Save the edited sequence (with augmentation)",
        "'c' - Cancel recording/editing",
        "'p' - Process all gestures for training (augment & extract features)",
        "'h' - Show/hide help window",
        "'l' - List saved gestures",
        "'ESC' - Exit the program"
    ]

    for i, control in enumerate(controls):
        cv.putText(help_img, control, (40, 100 + i * 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Recording & editing workflow
    cv.putText(help_img, "Recording & Editing Workflow:", (20, 380),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 1, cv.LINE_AA)

    workflow = [
        "1. Press 'r' to start recording a new gesture",
        "2. Perform the gesture with clear movements",
        "3. Press 's' to stop recording and prepare for editing",
        "4. Use the terminal interface to edit the sequence:",
        "   - Remove unwanted states",
        "   - Rearrange states if needed",
        "5. Press 's' in the terminal to save your changes",
        "6. Press 'a' in the main window to save the gesture"
    ]

    for i, step in enumerate(workflow):
        cv.putText(help_img, step, (40, 410 + i * 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Data augmentation info
    cv.putText(help_img, "Data Augmentation:", (20, 640),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 200), 1, cv.LINE_AA)

    augmentation = [
        "• Each saved gesture generates 5 augmented versions automatically",
        "• Augmentations include time stretching, jitter, and state modifications",
        "• Features are automatically extracted for model training",
        "• Use 'p' to process all gestures into a combined training dataset"
    ]

    for i, item in enumerate(augmentation):
        cv.putText(help_img, item, (40, 670 + i * 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Display help window
    cv.imshow("Help - FSL Recorder with Augmentation", help_img)
    cv.waitKey(0)
    cv.destroyWindow("Help - FSL Recorder with Augmentation")


def show_gestures_list_window(recorder):
    """Show a window listing all recorded gestures"""
    # Get saved gestures
    gestures = recorder.recorded_gestures

    if not gestures:
        list_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv.putText(list_img, "No gestures have been recorded yet", (20, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(list_img, "Press any key to close", (120, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)

        cv.imshow("Recorded Gestures", list_img)
        cv.waitKey(0)
        cv.destroyWindow("Recorded Gestures")
        return

    # Create image large enough for all gestures
    img_height = max(500, len(gestures) * 50 + 100)
    list_img = np.zeros((img_height, 700, 3), dtype=np.uint8)

    # Title
    cv.putText(list_img, f"Recorded Gestures ({len(gestures)})", (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    # Column headers
    cv.putText(list_img, "Name", (20, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv.LINE_AA)
    cv.putText(list_img, "Description", (200, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv.LINE_AA)
    cv.putText(list_img, "Components", (450, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv.LINE_AA)

    # Draw line
    cv.line(list_img, (10, 75), (690, 75), (100, 100, 100), 1)

    # List gestures
    y_pos = 100
    for name, info in gestures.items():
        # Name
        cv.putText(list_img, name, (20, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # Description
        description = info.get("description", "")
        if len(description) > 25:  # Truncate long descriptions
            description = description[:22] + "..."
        cv.putText(list_img, description, (200, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)

        # Component count
        components = info.get("simplified_sequence", [])
        cv.putText(list_img, f"{len(components)} states", (450, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv.LINE_AA)

        y_pos += 30

    # Footer
    cv.putText(list_img, "Press any key to close", (250, y_pos + 50),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 1, cv.LINE_AA)

    # Display list window
    cv.imshow("Recorded Gestures", list_img)
    cv.waitKey(0)
    cv.destroyWindow("Recorded Gestures")


def prompt_for_gesture_name():
    """Show a prompt to enter gesture name and description"""
    prompt_img = np.zeros((300, 500, 3), dtype=np.uint8)

    # Title
    cv.putText(prompt_img, "New Gesture Recording", (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    # Instructions
    cv.putText(prompt_img, "Enter gesture name in the terminal", (20, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1, cv.LINE_AA)

    cv.putText(prompt_img, "Example names:", (20, 120),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    examples = [
        "good_morning",
        "thank_you",
        "hello",
        "goodbye"
    ]

    for i, example in enumerate(examples):
        cv.putText(prompt_img, example, (40, 150 + i * 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv.LINE_AA)

    # Footer
    cv.putText(prompt_img, "Check the terminal window now", (120, 270),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 1, cv.LINE_AA)

    # Display prompt window
    cv.imshow("Enter Gesture Name", prompt_img)

    # Get input from terminal
    print("\nEnter name for the new gesture (e.g., good_morning, thank_you): ")
    name = input().strip()

    print("Enter a brief description (optional): ")
    description = input().strip()

    cv.destroyWindow("Enter Gesture Name")

    return name, description


def load_labels():
    """Load the keypoint classifier labels"""
    labels = []

    if os.path.exists(KEYPOINT_CLASSIFIER_LABEL):
        try:
            with open(KEYPOINT_CLASSIFIER_LABEL, encoding='utf-8-sig') as f:
                for line in f:
                    labels.append(line.strip())
            print(f"Loaded {len(labels)} labels")
        except Exception as e:
            print(f"Error loading labels: {e}")

    if not labels:
        # Default alphabet labels if file not found
        labels = [str(i) for i in range(10)] + [chr(i + 97) for i in range(26)]
        print("Using default labels")

    return labels


def main():
    # Parse arguments
    args = get_args()

    # Camera setup
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=args.model_selection,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Initialize keypoint classifier
    keypoint_classifier = KeyPointClassifier()

    # Load labels
    keypoint_classifier_labels = load_labels()

    # FPS Calculation
    fps_calc = CvFpsCalc(buffer_len=10)

    # Initialize prediction history for stability
    prediction_history = deque(maxlen=5)

    # Initialize gesture recorder with specified augmentation factor
    gesture_recorder = GestureRecorder()
    gesture_recorder.augmenter.augmentation_factor = args.augmentation_factor

    # Track wrist positions for movement detection (separate for each hand)
    wrist_positions_right = deque(maxlen=30)
    wrist_positions_left = deque(maxlen=30)

    # Component history
    component_history = deque(maxlen=100)

    # Track most recent components for state change detection (per hand)
    last_component_right = None
    last_component_left = None

    # Show help at startup
    show_help_window()

    # Print instructions
    print("\nFSL Clean Sequence Recorder with Data Augmentation")
    print("==============================================")
    print("Captures clean gesture sequences with smart filtering, editing, and data augmentation")
    print("\nControls:")
    print("  'r' - Start recording a new gesture")
    print("  's' - Stop recording and prepare for editing")
    print("  'e' - Edit the recorded sequence")
    print("  'a' - Save the edited sequence (with augmentation)")
    print("  'c' - Cancel recording/editing")
    print("  'p' - Process all gestures for training (augment & extract features)")
    print("  'l' - List saved gestures")
    print("  'h' - Show/hide help window")
    print("  'ESC' - Exit program")
    print(f"\nAugmentation factor: {args.augmentation_factor} (variants per gesture)")

    # Main loop
    while True:
        # FPS calculation
        fps = fps_calc.get()

        # Key handling
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == ord('h'):  # Show help
            show_help_window()
        elif key == ord('l'):  # List gestures
            show_gestures_list_window(gesture_recorder)
        elif key == ord('r') and not gesture_recorder.is_recording():  # Start recording
            if hasattr(gesture_recorder, 'finalized_sequence'):
                print("You have an unsaved sequence. Press 'a' to save it or 'c' to discard it.")
            else:
                name, description = prompt_for_gesture_name()
                if name:
                    gesture_recorder.start_recording(name, description)
        elif key == ord('s') and gesture_recorder.is_recording():  # Stop recording
            finalized = gesture_recorder.stop_recording()
            if finalized:
                print(f"Recording stopped with {len(finalized)} sequence states.")
                print("Press 'e' to edit the sequence, 'a' to save it, or 'c' to discard it.")
        elif key == ord('e') and hasattr(gesture_recorder, 'finalized_sequence'):  # Edit sequence
            gesture_recorder.edit_sequence()
        elif key == ord('a') and hasattr(gesture_recorder, 'finalized_sequence'):  # Save sequence
            saved = gesture_recorder.save_recording(perform_augmentation=True)
            if saved:
                print(f"Gesture '{gesture_recorder.gesture_name}' saved successfully with augmentation.")
        elif key == ord('p'):  # Process all gestures for training
            if gesture_recorder.process_all_for_training():
                print("All gestures have been processed for training.")
        elif key == ord('c'):  # Cancel recording/editing
            if gesture_recorder.is_recording():
                gesture_recorder.cancel_recording()
                last_component_right = None
                last_component_left = None
            elif hasattr(gesture_recorder, 'finalized_sequence'):
                delattr(gesture_recorder, 'finalized_sequence')
                print("Sequence editing canceled.")

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Color conversion for MediaPipe
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Display FPS
        cv.putText(debug_image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(debug_image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        # Draw recording status
        debug_image = draw_recording_status(debug_image, gesture_recorder)

        # Process hand landmarks
        if results.multi_hand_landmarks:
            # Process all detected hands
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):

                # Get handedness (left or right hand)
                hand_type = handedness.classification[0].label  # "Left" or "Right"

                # Calculate bounding rectangle
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                # Extract landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Store wrist position based on handedness
                if landmark_list:
                    if hand_type == "Right":
                        wrist_positions_right.append(landmark_list[0])
                    else:
                        wrist_positions_left.append(landmark_list[0])

                # Pre-process landmarks
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Hand sign classification
                hand_sign_id = -1
                confidence = 0.0

                if keypoint_classifier.model_loaded:
                    # Get raw prediction
                    hand_sign_id, confidence, result_array = keypoint_classifier(pre_processed_landmark_list)

                    # Improve prediction particularly for O/C
                    hand_sign_id, confidence = improve_prediction(
                        hand_sign_id, confidence, result_array, landmark_list, keypoint_classifier_labels)

                    # Add to prediction history
                    prediction_history.append(hand_sign_id)

                    # Get most common prediction for stability
                    counter = Counter(prediction_history)
                    most_common = counter.most_common(1)
                    if most_common:
                        hand_sign_id = most_common[0][0]

                # Get handshape text (number or letter)
                handshape_text = ""
                if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                    handshape_text = keypoint_classifier_labels[hand_sign_id].lower()

                # Detect simple orientation
                orientation = detect_orientation(landmark_list, hand_type)

                # Detect movement based on hand type
                movement = "stationary"
                if hand_type == "Right" and len(wrist_positions_right) > 5:
                    movement = detect_simplified_movement(wrist_positions_right)
                elif hand_type == "Left" and len(wrist_positions_left) > 5:
                    movement = detect_simplified_movement(wrist_positions_left)

                # Draw landmarks with handedness coloring
                debug_image = draw_landmarks(debug_image, landmark_list, hand_type)

                # Display hand sign, orientation and movement
                debug_image = draw_info_text(
                    debug_image,
                    handshape_text.upper(),
                    orientation,
                    movement,
                    brect,
                    confidence,
                    hand_type
                )

                # Create current component
                current_component = {
                    "handshape": handshape_text,
                    "orientation": orientation,
                    "movement": movement,
                    "timestamp": time.time(),
                    "confidence": confidence,
                    "handedness": hand_type
                }

                # Check if component changed from last frame (per hand)
                component_changed = False
                if hand_type == "Right":
                    if last_component_right is None:
                        component_changed = True
                    else:
                        if (last_component_right["handshape"] != current_component["handshape"] or
                                last_component_right["orientation"] != current_component["orientation"] or
                                last_component_right["movement"] != current_component["movement"]):
                            component_changed = True

                    if component_changed:
                        component_history.append(current_component)
                        last_component_right = current_component
                else:  # Left hand
                    if last_component_left is None:
                        component_changed = True
                    else:
                        if (last_component_left["handshape"] != current_component["handshape"] or
                                last_component_left["orientation"] != current_component["orientation"] or
                                last_component_left["movement"] != current_component["movement"]):
                            component_changed = True

                    if component_changed:
                        component_history.append(current_component)
                        last_component_left = current_component

                # If recording, add frame
                if gesture_recorder.is_recording():
                    gesture_recorder.add_frame(
                        handshape_text,
                        orientation,
                        movement,
                        confidence,
                        hand_type
                    )

        # Draw component history visualization
        debug_image = draw_component_history(debug_image, component_history)

        # Draw finalized sequence if available
        if hasattr(gesture_recorder, 'finalized_sequence'):
            debug_image = draw_sequence_visualization(debug_image, gesture_recorder.finalized_sequence)

        # Draw augmentation info
        debug_image = draw_augmentation_info(debug_image)

        # Display the image
        cv.imshow('FSL Clean Sequence Recorder with Data Augmentation', debug_image)

    # Cleanup
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()