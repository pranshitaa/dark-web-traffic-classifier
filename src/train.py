import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Ensure the root project directory is in the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import TrafficDataPreprocessor

def create_synthetic_data(file_path):
    """Generates synthetic traffic dataset if none exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.random.seed(42)
    n_samples = 200
    
    # Normal traffic: shorter duration, larger packets, low inter-arrival time, high byte count
    n_normal = n_samples // 2
    normal_duration = np.random.uniform(0.5, 50.0, n_normal)
    normal_packet_size = np.random.uniform(1100.0, 1500.0, n_normal)
    normal_iat = np.random.uniform(0.001, 0.05, n_normal)
    normal_bytes = normal_packet_size * np.random.uniform(20, 200, n_normal)
    normal_labels = ['Normal'] * n_normal
    
    # Dark Web traffic: longer burst duration, smaller/fixed padded packets, high jitter/iat
    n_dark = n_samples - n_normal
    dark_duration = np.random.uniform(60.0, 300.0, n_dark)
    dark_packet_size = np.random.uniform(250.0, 600.0, n_dark)
    dark_iat = np.random.uniform(0.3, 1.5, n_dark)
    dark_bytes = dark_packet_size * np.random.uniform(50, 150, n_dark)
    dark_labels = ['Dark Web'] * n_dark
    
    df = pd.DataFrame({
        'flow_duration': np.concatenate([normal_duration, dark_duration]),
        'packet_size': np.concatenate([normal_packet_size, dark_packet_size]),
        'inter_arrival_time': np.concatenate([normal_iat, dark_iat]),
        'byte_count': np.concatenate([normal_bytes, dark_bytes]),
        'label': normal_labels + dark_labels
    }).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    df.to_csv(file_path, index=False)
    print(f"Created synthetic sample dataset at: {file_path}")
    return df

def train_sklearn(data_path="data/sample_traffic.csv", model_type="sklearn", output_dir="models"):
    """
    Trains a Scikit-Learn or Neural Network classifier and saves the model + preprocessor.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Creating sample traffic data...")
        df = create_synthetic_data(data_path)
    else:
        df = pd.read_csv(data_path)
        
    print(f"Loaded dataset: {len(df)} records from '{data_path}'")
    
    # 2. Preprocess
    preprocessor = TrafficDataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    if 'label' not in df_clean.columns:
        raise ValueError("Dataset must contain a 'label' column ('Normal' or 'Dark Web').")
        
    y = preprocessor.encode_labels(df_clean['label'])
    X = preprocessor.normalize_features(df_clean, training=True)
    
    # 3. Train/Test Split
    # Only stratify if dataset has sufficient samples for both classes in test split
    class_counts = pd.Series(y).value_counts()
    min_class_count = class_counts.min() if len(class_counts) > 0 else 0
    use_stratify = y if (len(y) >= 10 and min_class_count >= 2) else None
    test_size = 0.2 if len(y) >= 5 else 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=use_stratify
    )
    
    # 4. Model Selection
    model_name = "trained_model.pkl"
    if model_type in ['dl', 'neural_network', 'mlp']:
        print("Training Deep Neural Network (MLP Classifier)...")
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=400,
            random_state=42,
            early_stopping=True
        )
        model_name = "dl_model.pkl"
    elif model_type == 'svm':
        print("Training Support Vector Classifier (SVM)...")
        model = SVC(kernel='rbf', probability=True, random_state=42)
    else:
        print("Training Random Forest Classifier with Probability Calibration (Platt Scaling)...")
        base_rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=42)
        # Use CalibratedClassifierCV if sufficient samples exist, otherwise base_rf
        if len(X_train) >= 9 and len(np.unique(y_train)) > 1:
            model = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=3)
        else:
            model = base_rf
        
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*45)
    print("           MODEL EVALUATION RESULTS")
    print("="*45)
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("="*45)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Dark Web']))
    
    # 6. Save Artifacts
    model_path = os.path.join(output_dir, model_name)
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    preprocessor.save(preprocessor_path)
    
    # Also save as default trained_model.pkl if this is the primary run
    default_model_path = os.path.join(output_dir, "trained_model.pkl")
    if model_path != default_model_path and not os.path.exists(default_model_path):
        with open(default_model_path, 'wb') as f:
            pickle.dump(model, f)
            
    print(f"\n[+] Saved trained model to: '{model_path}'")
    print(f"[+] Saved preprocessor to : '{preprocessor_path}'")
    return model, preprocessor

def main():
    parser = argparse.ArgumentParser(description="Dark Web Traffic Classifier Training Pipeline")
    parser.add_argument('--model_type', type=str, default='sklearn', choices=['sklearn', 'rf', 'svm', 'dl', 'mlp'],
                        help="Model architecture: 'sklearn' (Random Forest), 'svm', or 'dl' (Neural Network)")
    parser.add_argument('--data', type=str, default='data/sample_traffic.csv',
                        help="Path to training CSV dataset")
    parser.add_argument('--output_dir', type=str, default='models',
                        help="Directory to save model and preprocessor artifacts")
    
    args = parser.parse_args()
    train_sklearn(data_path=args.data, model_type=args.model_type, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
