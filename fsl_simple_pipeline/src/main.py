import os
import numpy as np
import argparse
from data_collection.collector import GestureCollector
from data_collection.dataset_loader import DatasetLoader
from feature_extraction.extractor import FeatureExtractor
from preprocessing.preprocessor import Preprocessor
from models.gesture_model import GestureModel
from realtime.recognizer import GestureRecognizer

def train_model(args):
    """Train the gesture recognition model."""
    print("Loading dataset...")
    dataset_loader = DatasetLoader(dataset_path=args.dataset_path)
    
    # Load training data
    X_train, y_train = dataset_loader.load_dataset(split="train", max_samples=args.max_samples)
    
    # Load validation data
    X_val, y_val = dataset_loader.load_dataset(split="test", max_samples=args.max_samples // 4)
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    preprocessor = Preprocessor()
    
    # Process training data
    print("Processing training data...")
    X_train_processed = []
    for landmarks in X_train:
        # Preprocess
        processed = preprocessor.preprocess(landmarks)
        
        # Extract features
        features = feature_extractor.extract_features(processed)
        
        # Combine features
        combined_features = np.concatenate([
            features['handshape'],
            features['orientation'],
            features['location'],
            features['movement']
        ], axis=1)
        
        X_train_processed.append(combined_features)
    
    X_train_processed = np.array(X_train_processed)
    
    # Process validation data
    print("Processing validation data...")
    X_val_processed = []
    for landmarks in X_val:
        # Preprocess
        processed = preprocessor.preprocess(landmarks)
        
        # Extract features
        features = feature_extractor.extract_features(processed)
        
        # Combine features
        combined_features = np.concatenate([
            features['handshape'],
            features['orientation'],
            features['location'],
            features['movement']
        ], axis=1)
        
        X_val_processed.append(combined_features)
    
    X_val_processed = np.array(X_val_processed)
    
    # Convert labels to one-hot encoding
    num_classes = len(dataset_loader.get_label_mapping())
    y_train_one_hot = np.eye(num_classes)[y_train]
    y_val_one_hot = np.eye(num_classes)[y_val]
    
    # Initialize and train model
    input_shape = (X_train_processed.shape[1], X_train_processed.shape[2])
    model = GestureModel(input_shape=input_shape, num_classes=num_classes)
    
    print("Training model...")
    history = model.train(
        X_train_processed, y_train_one_hot,
        X_val_processed, y_val_one_hot,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Save the model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "gesture_model")
    model.save(model_path)
    
    print(f"\nTraining completed!")
    print(f"Model saved to '{model_path}'")
    
    return model_path, dataset_loader.get_index_mapping()

def recognize_realtime(args):
    """Perform real-time gesture recognition."""
    # Load model and label mapping
    model_path = args.model_path
    label_mapping = args.label_mapping
    
    # Initialize recognizer
    recognizer = GestureRecognizer(
        model_path=model_path,
        label_mapping=label_mapping,
        output_dir=args.output_dir
    )
    
    # Start real-time recognition
    recognizer.recognize_realtime()

def collect_data(args):
    """Collect gesture data manually."""
    collector = GestureCollector(output_dir=args.output_dir)
    
    # Collect gestures
    for gesture in args.gestures:
        print(f"\nCollecting gesture: {gesture}")
        collector.collect_gesture(gesture, num_frames=args.num_frames)

def main():
    parser = argparse.ArgumentParser(description="FSL Gesture Recognition Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--dataset-path", type=str, default="FSL SLR Dataset", help="Path to the dataset")
    train_parser.add_argument("--model-dir", type=str, default="models", help="Directory to save the model")
    train_parser.add_argument("--max-samples", type=int, default=100, help="Maximum number of samples to use")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    # Recognize command
    recognize_parser = subparsers.add_parser("recognize", help="Perform real-time recognition")
    recognize_parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    recognize_parser.add_argument("--label-mapping", type=str, required=True, help="Path to the label mapping file")
    recognize_parser.add_argument("--output-dir", type=str, default="collected_data", help="Directory to save collected data")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect gesture data")
    collect_parser.add_argument("--gestures", type=str, nargs="+", required=True, help="Gestures to collect")
    collect_parser.add_argument("--num-frames", type=int, default=30, help="Number of frames to collect per gesture")
    collect_parser.add_argument("--output-dir", type=str, default="collected_data", help="Directory to save collected data")
    
    args = parser.parse_args()
    
    if args.command == "train":
        model_path, label_mapping = train_model(args)
        
        # Save label mapping
        os.makedirs("models", exist_ok=True)
        with open("models/label_mapping.txt", "w") as f:
            for idx, label in label_mapping.items():
                f.write(f"{idx}\t{label}\n")
        
        print("Label mapping saved to 'models/label_mapping.txt'")
    
    elif args.command == "recognize":
        # Load label mapping
        label_mapping = {}
        with open(args.label_mapping, "r") as f:
            for line in f:
                idx, label = line.strip().split("\t")
                label_mapping[int(idx)] = label
        
        args.label_mapping = label_mapping
        recognize_realtime(args)
    
    elif args.command == "collect":
        collect_data(args)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 