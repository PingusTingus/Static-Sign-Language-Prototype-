#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSL Gesture Sentence Recognition System with Text-to-Speech
Recognizes gestures and combines them into sentences with speech output

Created by: PingusTingus
Date: 2025-04-15 07:15:21
"""

import os
import sys
import csv
import time
import copy
import json
import itertools
import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
from collections import Counter, deque
from datetime import datetime
import threading
import pickle

# Import text-to-speech library (use pyttsx3 which is offline and works cross-platform)
try:
    import pyttsx3

    TTS_ENABLED = True
except ImportError:
    print("Warning: pyttsx3 not installed. Text-to-speech will be disabled.")
    print("To enable text-to-speech, install pyttsx3 with: pip install pyttsx3")
    TTS_ENABLED = False

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
KEYPOINT_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'keypoint_classifier', 'keypoint_classifier.tflite')
KEYPOINT_CLASSIFIER_LABEL = os.path.join(MODEL_DIR, 'keypoint_classifier', 'keypoint_classifier_label.csv')
RECORDED_GESTURES_DIR = os.path.join(DATA_DIR, 'recorded_gestures')
TRAINED_MODEL_DIR = os.path.join(DATA_DIR, 'trained_models')
SENTENCES_LOG_PATH = os.path.join(DATA_DIR, 'sentences_log.json')

# Create directories if they don't exist
os.makedirs(os.path.join(MODEL_DIR, 'keypoint_classifier'), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RECORDED_GESTURES_DIR, exist_ok=True)
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

# Constants
MAX_SEQUENCE_LENGTH = 10  # Maximum number of states in a sequence
PADDING_VALUE = -1  # Value used for padding sequences
STATIONARY_THRESHOLD = 7  # Frames to consider stationary
MIN_GESTURE_LENGTH = 2  # Minimum states for a valid gesture
STATIONARY_TIMEOUT = 1.0  # Seconds to wait after stationary before prediction
NO_HANDS_TIMEOUT = 3.0  # Seconds without hands to complete sentence

# Print execution info
print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: PingusTingus")


class TextToSpeech:
    """Text-to-speech engine for speaking sentences"""

    def __init__(self):
        self.enabled = TTS_ENABLED
        if not self.enabled:
            return

        # Initialize the TTS engine
        try:
            self.engine = pyttsx3.init()

            # Set properties
            self.engine.setProperty('rate', 150)  # Speed (words per minute)
            self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

            # Try to get a better voice if available
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice (usually clearer for many systems)
                female_voices = [voice for voice in voices if 'female' in voice.name.lower()]
                if female_voices:
                    self.engine.setProperty('voice', female_voices[0].id)

            # For background speaking
            self.speech_thread = None
            self.is_speaking = False

        except Exception as e:
            print(f"Error initializing text-to-speech: {e}")
            self.enabled = False

    def speak(self, text, blocking=False):
        """Speak the given text"""
        if not self.enabled or not text:
            return False

        # If already speaking, stop current speech
        if hasattr(self, 'is_speaking') and self.is_speaking:
            self.stop()

        try:
            if blocking:
                # Speak synchronously
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                # Speak asynchronously in a background thread
                self.speech_thread = threading.Thread(target=self._speak_in_background, args=(text,))
                self.speech_thread.daemon = True
                self.speech_thread.start()
                self.is_speaking = True

            return True
        except Exception as e:
            print(f"Error speaking text: {e}")
            return False

    def _speak_in_background(self, text):
        """Helper function for background speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in background speech: {e}")
        finally:
            self.is_speaking = False

    def stop(self):
        """Stop any ongoing speech"""
        if not self.enabled:
            return

        if hasattr(self, 'is_speaking') and self.is_speaking:
            try:
                self.engine.stop()
                self.is_speaking = False
            except:
                pass


class SentenceManager:
    """Manages sentence construction from detected gestures"""

    def __init__(self):
        # Initialize sentence components
        self.current_sentence = []
        self.last_gesture = None
        self.last_gesture_time = 0
        self.confidence_threshold = 0.5
        self.tts_engine = TextToSpeech()
        self.last_spoken_sentence = ""
        self.sentences_log = self._load_sentences_log()
        self.no_hands_start_time = None

    def _load_sentences_log(self):
        """Load previous sentences log if available"""
        if os.path.exists(SENTENCES_LOG_PATH):
            try:
                with open(SENTENCES_LOG_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading sentences log: {e}")

        # Create a new log if none exists
        return {
            "sentences": [],
            "session_start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_count": 0
        }

    def _save_sentences_log(self):
        """Save sentences log to file"""
        with open(SENTENCES_LOG_PATH, 'w') as f:
            json.dump(self.sentences_log, f, indent=2)

    def add_gesture(self, gesture_name, confidence):
        """Add a new gesture to the current sentence if it meets criteria"""
        current_time = time.time()

        # Check if the gesture meets our confidence threshold
        if confidence < self.confidence_threshold:
            print(f"Low confidence detection: {gesture_name} ({confidence:.2f})")
            return False

        # Avoid rapid duplicate detections (within 1 second)
        if gesture_name == self.last_gesture and current_time - self.last_gesture_time < 1.0:
            return False

        # Update last gesture tracking
        self.last_gesture = gesture_name
        self.last_gesture_time = current_time

        # Add gesture to sentence
        self.current_sentence.append({
            "gesture": gesture_name,
            "confidence": confidence,
            "timestamp": current_time
        })

        print(f"Added to sentence: {gesture_name} ({confidence:.2f})")

        return True

    def get_current_text(self):
        """Convert current gesture sequence to readable text"""
        if not self.current_sentence:
            return ""

        # Simple conversion - just join the gestures with spaces
        # In a real system, you might want to add grammar rules
        text = " ".join(item["gesture"] for item in self.current_sentence)

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        return text

    def complete_sentence(self):
        """Complete the current sentence and speak it"""
        if not self.current_sentence:
            return ""

        # Generate the sentence text
        sentence_text = self.get_current_text()

        # Speak the sentence
        self.tts_engine.speak(sentence_text, blocking=False)
        self.last_spoken_sentence = sentence_text

        # Log the sentence
        self._log_sentence(sentence_text)

        # Clear current sentence
        self.current_sentence = []

        return sentence_text

    def _log_sentence(self, text):
        """Add a completed sentence to the log"""
        if not text:
            return

        # Create log entry
        entry = {
            "text": text,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gesture_count": len(self.current_sentence)
        }

        # Add to log
        self.sentences_log["sentences"].append(entry)
        self.sentences_log["total_count"] += 1

        # Save the updated log
        self._save_sentences_log()

    def clear_sentence(self):
        """Clear the current sentence without completing it"""
        self.current_sentence = []
        self.last_gesture = None

    def handle_no_hands(self, hands_detected):
        """Handle the case when no hands are detected to auto-complete sentences"""
        current_time = time.time()

        if not hands_detected:
            # Start/continue tracking time without hands
            if self.no_hands_start_time is None:
                self.no_hands_start_time = current_time

            # Check if we've passed the timeout
            elif current_time - self.no_hands_start_time >= NO_HANDS_TIMEOUT:
                # Complete the sentence if we have one
                if self.current_sentence:
                    self.complete_sentence()
                self.no_hands_start_time = None  # Reset timer
        else:
            # Reset the timer when hands are detected
            self.no_hands_start_time = None

    def get_completion_status(self):
        """Get status of sentence completion for display"""
        if self.no_hands_start_time is None:
            return 0.0

        elapsed = time.time() - self.no_hands_start_time
        if elapsed >= NO_HANDS_TIMEOUT:
            return 1.0

        return elapsed / NO_HANDS_TIMEOUT


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


class GestureRecognizer:
    """Recognizer for sign language gestures with sequence model"""

    def __init__(self, model_type='lstm'):
        # Paths for the sequence model
        self.model_type = model_type
        self.model_path = os.path.join(TRAINED_MODEL_DIR, f'gesture_model_{model_type}_final.h5')
        self.encoding_path = os.path.join(TRAINED_MODEL_DIR, 'encoding_dicts.pkl')
        self.label_encoder_path = os.path.join(TRAINED_MODEL_DIR, 'label_encoder.pkl')

        # State tracking
        self.current_sequence = []
        self.last_prediction = None
        self.state = "WAITING"  # WAITING -> RECORDING -> PROCESSING -> WAITING
        self.stationary_counter = 0
        self.last_component = None
        self.recording_start_time = 0
        self.stationary_time = 0

        # Sentence manager for building sentences
        self.sentence_manager = SentenceManager()

        # Load model components
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the gesture recognition model and related files"""
        # Check if files exist
        if not all(os.path.exists(p) for p in [self.model_path, self.encoding_path, self.label_encoder_path]):
            print("Some model files are missing. Recognition will be disabled.")
            print(f"Expected model at: {self.model_path}")
            print(f"Expected encodings at: {self.encoding_path}")
            print(f"Expected label encoder at: {self.label_encoder_path}")
            return False

        try:
            # Load model
            self.model = tf.keras.models.load_model(self.model_path)

            # Load encodings
            with open(self.encoding_path, 'rb') as f:
                self.encodings = pickle.load(f)

            # Check encoding format
            if not all(k in self.encodings for k in ['handshape_dict', 'orientation_dict', 'movement_dict']):
                print(f"Invalid encoding format. Keys: {self.encodings.keys()}")
                return False

            # Load label encoder
            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

            self.model_loaded = True
            print(f"Model loaded successfully!")
            print(f"Available gestures: {', '.join(self.label_encoder.classes_)}")
            return True

        except Exception as e:
            print(f"Error loading model components: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update(self, handshape, orientation, movement, confidence, is_stationary):
        """Update recognizer state with new components"""
        if not self.model_loaded:
            return self.state

        # Check if component is valid
        if not handshape or not orientation:  # Movement can be None for transition
            return self.state

        current_time = time.time()

        # Create current component tuple for comparison
        current_component = (handshape.lower(), orientation, movement)

        # State machine
        if self.state == "WAITING":
            # In waiting state, look for stationary pose to start recording
            if is_stationary:
                self.stationary_counter += 1
                if self.stationary_counter >= STATIONARY_THRESHOLD:
                    self.state = "RECORDING"
                    self.current_sequence = []  # Clear sequence
                    self.recording_start_time = current_time
                    print("Detected stationary pose - started recording")
            else:
                self.stationary_counter = 0

        elif self.state == "RECORDING":
            # Recording state - collecting gesture components

            # Check if different from last component (avoid repeating)
            is_different = (current_component != self.last_component)

            # Add component to sequence if it's different
            if is_different and self.last_component is not None:
                # Calculate duration of previous component
                duration = 0.2  # Default minimum duration
                if self.last_component is not None:
                    duration = max(duration, 0.2)

                # Add the previous component since it's complete
                self._add_component_to_sequence(
                    self.last_component[0],  # handshape
                    self.last_component[1],  # orientation
                    self.last_component[2],  # movement
                    confidence,
                    duration
                )

            # Update last component
            self.last_component = current_component

            # Check for end of gesture (stationary pose)
            if is_stationary:
                self.stationary_counter += 1
                if self.stationary_counter >= STATIONARY_THRESHOLD:
                    # If enough stationary frames, prepare to end recording
                    if self.stationary_time == 0:
                        self.stationary_time = current_time

                    # Check if stationary for long enough
                    if current_time - self.stationary_time >= STATIONARY_TIMEOUT:
                        # Add final stationary component if different
                        if is_different:
                            self._add_component_to_sequence(
                                handshape.lower(),
                                orientation,
                                "stationary",  # Force stationary as final state
                                confidence,
                                STATIONARY_TIMEOUT
                            )

                        self.state = "PROCESSING"
                        print(f"Detected end of gesture - sequence has {len(self.current_sequence)} states")

                        # Process the sequence and get prediction
                        prediction = self._process_sequence()

                        # Add to sentence if the prediction is valid
                        if prediction:
                            gesture_name, confidence = prediction
                            self.sentence_manager.add_gesture(gesture_name, confidence)
            else:
                self.stationary_counter = 0
                self.stationary_time = 0

        elif self.state == "PROCESSING":
            # After prediction, wait a bit then go back to waiting state
            if current_time - self.stationary_time >= STATIONARY_TIMEOUT:
                self.state = "WAITING"
                self.stationary_counter = 0
                self.stationary_time = 0
                self.last_component = None

        return self.state

    def _add_component_to_sequence(self, handshape, orientation, movement, confidence, duration):
        """Add a component to the sequence"""
        # Create state data in same format as training
        state_data = {
            "handshape": handshape,
            "orientation": orientation,
            "movement": movement,
            "confidence": confidence,
            "duration": duration
        }

        # Add to sequence (avoid duplicate consecutive states)
        if not self.current_sequence or (
                self.current_sequence[-1]["handshape"] != handshape or
                self.current_sequence[-1]["orientation"] != orientation or
                self.current_sequence[-1]["movement"] != movement
        ):
            self.current_sequence.append(state_data)

        # Keep only the most recent states
        if len(self.current_sequence) > MAX_SEQUENCE_LENGTH:
            self.current_sequence = self.current_sequence[-MAX_SEQUENCE_LENGTH:]

    def _process_sequence(self):
        """Process the completed sequence and make prediction"""
        if len(self.current_sequence) < MIN_GESTURE_LENGTH:
            print(f"Sequence too short ({len(self.current_sequence)} states) - skipping prediction")
            self.last_prediction = None
            return None

        try:
            # Preprocess sequence using the standardized preprocessing function
            X = self._preprocess_sequence(self.current_sequence)

            # Make prediction
            prediction = self.model.predict(X, verbose=0)

            # Get highest probability class
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]

            # Convert to gesture name
            gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]

            # Store prediction
            self.last_prediction = (gesture_name, confidence)

            # Print prediction details
            print(f"Predicted gesture: {gesture_name} with confidence {confidence:.4f}")

            # Show top probabilities
            top_indices = np.argsort(prediction[0])[-3:][::-1]
            print("Top 3 predictions:")
            for idx in top_indices:
                if idx < len(self.label_encoder.classes_):
                    class_name = self.label_encoder.inverse_transform([idx])[0]
                    prob = prediction[0][idx]
                    print(f"  {class_name}: {prob:.4f}")

            return self.last_prediction

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            self.last_prediction = None
            return None

    def _preprocess_sequence(self, sequence):
        """
        Convert a sequence to model input format
        Uses the same preprocessing as the training code
        """
        # Get dictionaries
        handshape_dict = self.encodings['handshape_dict']
        orientation_dict = self.encodings['orientation_dict']
        movement_dict = self.encodings['movement_dict']

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
            sequence_features = sequence_features[:MAX_SEQUENCE_LENGTH]
        elif len(sequence_features) < MAX_SEQUENCE_LENGTH:
            padding = [[PADDING_VALUE] * len(sequence_features[0]) for _ in
                       range(MAX_SEQUENCE_LENGTH - len(sequence_features))]
            sequence_features.extend(padding)

        # Convert to numpy array
        return np.array([sequence_features])

    def reset(self):
        """Reset the recognizer state"""
        self.current_sequence = []
        self.state = "WAITING"
        self.stationary_counter = 0
        self.last_component = None
        self.stationary_time = 0
        self.last_prediction = None
        print("Gesture recognizer reset")

    def get_current_sequence(self):
        """Get the current sequence"""
        return self.current_sequence

    def get_last_prediction(self):
        """Get the last predicted gesture"""
        return self.last_prediction

    def get_recording_duration(self):
        """Get the duration of the current recording"""
        if self.state == "RECORDING" and self.recording_start_time > 0:
            return time.time() - self.recording_start_time
        return 0

    def handle_no_hands_detection(self, hands_detected):
        """Handle tracking for no hands detection"""
        self.sentence_manager.handle_no_hands(hands_detected)

    def get_current_sentence(self):
        """Get the current sentence text"""
        return self.sentence_manager.get_current_text()

    def get_sentence_status(self):
        """Get sentence completion status for display"""
        return self.sentence_manager.get_completion_status()

    def complete_sentence(self):
        """Force completion of the current sentence"""
        return self.sentence_manager.complete_sentence()

    def clear_sentence(self):
        """Clear the current sentence without speaking"""
        self.sentence_manager.clear_sentence()


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
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--no_tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--no_hands_timeout", type=float, default=NO_HANDS_TIMEOUT,
                        help="Seconds without hands to trigger sentence completion")

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


def draw_prediction(image, prediction_result, y_position=70):
    """Draw prediction result prominently"""
    if not prediction_result:
        cv.putText(image, "Waiting for gesture...", (10, y_position),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv.LINE_AA)
        return image

    gesture_name, confidence = prediction_result

    # Draw a big attention-getting box
    bg_width = 400
    bg_height = 60
    cv.rectangle(image, (10, y_position - 40), (10 + bg_width, y_position + bg_height - 40),
                 (0, 100, 0), -1)

    # Draw the gesture name prominently
    cv.putText(image, f"GESTURE: {gesture_name.upper()}", (30, y_position),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    # Draw confidence
    cv.putText(image, f"Confidence: {confidence:.2f}", (30, y_position + 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1, cv.LINE_AA)

    return image


def draw_sentence_display(image, recognizer, start_y=150):
    """Draw the current sentence and completion status"""
    current_sentence = recognizer.get_current_sentence()
    completion_status = recognizer.get_sentence_status()

    # Calculate background dimensions
    bg_width = 900  # Wide enough for sentences
    bg_height = 120

    # Draw main background box
    cv.rectangle(image, (10, start_y), (10 + bg_width, start_y + bg_height),
                 (30, 40, 60), -1)

    # Title
    cv.putText(image, "Current Sentence:", (20, start_y + 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 1, cv.LINE_AA)

    # Display sentence or placeholder
    if current_sentence:
        # Handle long sentences by possibly splitting into two lines
        if len(current_sentence) > 70:
            # Find a good breaking point around the middle
            mid_point = len(current_sentence) // 2
            space_before = current_sentence.rfind(' ', 0, mid_point)
            space_after = current_sentence.find(' ', mid_point)

            # Choose the closest space
            if space_before != -1 and space_after != -1:
                split_point = space_before if mid_point - space_before < space_after - mid_point else space_after
            elif space_before != -1:
                split_point = space_before
            elif space_after != -1:
                split_point = space_after
            else:
                split_point = mid_point

            # Split the text
            first_half = current_sentence[:split_point]
            second_half = current_sentence[split_point:]

            # Draw the two lines
            cv.putText(image, first_half, (30, start_y + 65),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(image, second_half, (30, start_y + 95),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
        else:
            # Draw single line
            cv.putText(image, current_sentence, (30, start_y + 65),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
    else:
        # Placeholder when no sentence
        cv.putText(image, "< No gestures added yet >", (30, start_y + 65),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1, cv.LINE_AA)

    # Draw auto-completion progress bar if active
    if completion_status > 0:
        bar_width = bg_width - 40
        bar_height = 10
        bar_x = 20
        bar_y = start_y + bg_height - 20

        # Background bar
        cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (60, 60, 60), -1)

        # Progress fill
        fill_width = int(bar_width * completion_status)
        cv.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                     (0, 150, 255), -1)

        # Label
        cv.putText(image,
                   f"Auto-completing sentence in {NO_HANDS_TIMEOUT - completion_status * NO_HANDS_TIMEOUT:.1f}s...",
                   (bar_x + bar_width // 2 - 150, bar_y - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv.LINE_AA)

    # Draw controls
    controls_text = "SPACE: Complete sentence | BACKSPACE: Clear | S: Speak current sentence"
    cv.putText(image, controls_text, (20, start_y + bg_height - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 180, 255), 1, cv.LINE_AA)

    return image


def draw_sequence_status(image, sequence, max_display=5, start_y=300):
    """Draw the current gesture sequence"""
    if not sequence:
        return image

    cv.rectangle(image, (10, start_y - 10), (480, start_y + 30 * min(max_display, len(sequence))),
                 (60, 60, 60), -1)

    cv.putText(image, f"Current Sequence ({len(sequence)} states):", (15, start_y - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)

    display_items = sequence[-max_display:]

    for i, state in enumerate(display_items):
        y_pos = start_y + 20 * i

        handshape = state.get("handshape", "?")
        orientation = state.get("orientation", "?")
        movement = state.get("movement", "?")
        duration = state.get("duration", 0)

        component_text = f"[{handshape}, {orientation}, {movement}] ({duration:.1f}s)"
        cv.putText(image, component_text, (20, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_state_indicator(image, state, duration=0, is_stationary=False):
    """Draw the current state of the recognizer"""
    # Define colors for different states
    state_colors = {
        "WAITING": (100, 100, 100),  # Gray
        "RECORDING": (0, 0, 255),  # Red
        "PROCESSING": (0, 255, 0)  # Green
    }
    color = state_colors.get(state, (200, 200, 200))

    # Draw status box
    cv.rectangle(image, (10, 70), (300, 120), (40, 40, 40), -1)

    # Draw state text
    cv.putText(image, f"State: {state}", (20, 95),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)

    # Draw additional info based on state
    if state == "RECORDING":
        cv.putText(image, f"Recording: {duration:.1f}s", (20, 115),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)
    elif state == "WAITING":
        # Show stationary indicator
        msg = "Hold still to start"
        if is_stationary:
            msg = "Hand stationary - hold still to start"
        cv.putText(image, msg, (20, 115),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)
    elif state == "PROCESSING":
        cv.putText(image, "Processing gesture...", (20, 115),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

    return image


def draw_help(image, start_y=450):
    """Draw help text"""
    # Background
    cv.rectangle(image, (10, start_y), (300, start_y + 100), (40, 40, 40), -1)

    # Title
    cv.putText(image, "Controls:", (20, start_y + 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Control instructions
    controls = [
        "SPACE: Complete sentence",
        "BACKSPACE: Clear sentence",
        "S: Speak current sentence",
        "R: Reset recognizer",
        "Q/ESC: Quit"
    ]

    for i, control in enumerate(controls):
        cv.putText(image, control, (20, start_y + 40 + i * 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)

    return image


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

    # Update timeout
    global NO_HANDS_TIMEOUT
    NO_HANDS_TIMEOUT = args.no_hands_timeout

    # Initialize recognizer
    print("\n=== Initializing FSL Gesture Sentence System ===")
    recognizer = GestureRecognizer(model_type=args.model_type)

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
        static_image_mode=args.use_static_image_mode,
    )

    # Initialize keypoint classifier
    keypoint_classifier = KeyPointClassifier()

    # Load labels
    keypoint_classifier_labels = load_labels()

    # FPS Calculation
    fps_calc = CvFpsCalc(buffer_len=10)

    # Initialize prediction history for stability
    prediction_history = deque(maxlen=5)

    # Track wrist positions for movement detection (separate for each hand)
    wrist_positions_right = deque(maxlen=30)
    wrist_positions_left = deque(maxlen=30)

    # Print instructions
    print("\n=== FSL Gesture Sentence System ===")
    print("Create sentences by performing gestures one after another.")
    print("When you stop showing your hands, the sentence will be spoken.")
    print("\nControls:")
    print("  SPACE: Complete current sentence and speak it")
    print("  BACKSPACE: Clear current sentence")
    print("  S: Speak the current sentence")
    print("  R: Reset the recognizer")
    print("  Q/ESC: Quit")

    # Main loop
    hands_detected = False
    while True:
        # FPS calculation
        fps = fps_calc.get()

        # Key handling
        key = cv.waitKey(10)
        if key == 27 or key == ord('q'):  # ESC or q
            break
        elif key == ord('r'):  # Reset recognizer
            recognizer.reset()
        elif key == ord('s'):  # Speak current sentence
            sentence = recognizer.get_current_sentence()
            if sentence:
                print(f"Speaking: {sentence}")
                recognizer.complete_sentence()
        elif key == 8:  # Backspace - clear sentence
            recognizer.clear_sentence()
            print("Sentence cleared")
        elif key == 32:  # Space - complete sentence
            sentence = recognizer.complete_sentence()
            if sentence:
                print(f"Completed sentence: {sentence}")

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)  # Mirror display for more intuitive interaction
        debug_image = copy.deepcopy(image)

        # Color conversion for MediaPipe
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Reset hands detected flag
        hands_detected = False

        # Display FPS
        cv.putText(debug_image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(debug_image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 1, cv.LINE_AA)

        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            hands_detected = True

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

                # Update recognizer with components from first hand only
                # Only use the first detected hand for gesture recognition
                if hand_idx == 0 and handshape_text:
                    # Check if the hand is stationary for gesture recording
                    is_stationary = (movement == "stationary")
                    recognizer.update(handshape_text, orientation, movement, confidence, is_stationary)

        # Handle no-hands detection
        recognizer.handle_no_hands_detection(hands_detected)

        # Draw prediction
        prediction = recognizer.get_last_prediction()
        debug_image = draw_prediction(debug_image, prediction)

        # Draw state indicator - show if hand is stationary
        is_stationary = False
        if hands_detected and len(wrist_positions_right) > 5:
            # Check if the main hand is stationary
            movement = detect_simplified_movement(wrist_positions_right)
            is_stationary = (movement == "stationary")

        debug_image = draw_state_indicator(debug_image, recognizer.state,
                                           recognizer.get_recording_duration(),
                                           is_stationary)

        # Draw current sequence
        debug_image = draw_sequence_status(debug_image, recognizer.get_current_sequence())

        # Draw sentence display
        debug_image = draw_sentence_display(debug_image, recognizer)

        # Draw help
        debug_image = draw_help(debug_image)

        # Show the image
        cv.imshow('FSL Gesture Sentence Recognition', debug_image)

    # Cleanup
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()