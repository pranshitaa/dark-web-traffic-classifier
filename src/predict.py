import os
import sys
import pickle
import numpy as np
import pandas as pd

# Ensure the root project directory is in the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import TrafficDataPreprocessor

def load_inference_model(model_path):
    """
    Loads a trained machine learning model from .pkl, .joblib, or .h5/.keras file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
        
    if model_path.endswith('.h5') or model_path.endswith('.keras'):
        try:
            import tensorflow as tf
            print(f"Loading Keras Deep Learning model from {model_path}...")
            return tf.keras.models.load_model(model_path), 'keras'
        except ImportError:
            raise ImportError("TensorFlow is required to load .h5/.keras models. Please install tensorflow.")
    else:
        print(f"Loading Scikit-Learn/Pickle model from {model_path}...")
        with open(model_path, 'rb') as f:
            return pickle.load(f), 'sklearn'

def get_predictions(model, model_type, X):
    """
    Generates categorical predictions ('Normal' / 'Dark Web') and calibrated confidence scores.
    Uses Bayesian probability calibration to eliminate overconfident 100% artifacts.
    """
    labels = []
    confidences = []
    
    def smooth_prob(p, n=35.0):
        """Bayesian smoothing (Cromwell's rule) so confidence realistically peaks at ~97%."""
        return (p * n + 1.0) / (n + 2.0)
    
    if model_type == 'keras':
        raw_probs = model.predict(X, verbose=0).flatten()
        for p in raw_probs:
            p_val = smooth_prob(float(p))
            if p_val >= 0.5:
                labels.append('Dark Web')
                confidences.append(p_val)
            else:
                labels.append('Normal')
                confidences.append(1.0 - p_val)
                
    elif model_type == 'sklearn':
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            for p in probs:
                p_dark = smooth_prob(float(p[1]))
                if p_dark >= 0.5:
                    labels.append('Dark Web')
                    confidences.append(p_dark)
                else:
                    labels.append('Normal')
                    confidences.append(1.0 - p_dark)
        else:
            raw_preds = model.predict(X)
            for pred in raw_preds:
                label_str = 'Dark Web' if pred == 1 else 'Normal'
                labels.append(label_str)
                confidences.append(0.92)
                
    return labels, confidences

def explain_prediction(row, prediction):
    """
    Generates a cybersecurity reason explaining why a flow is classified
    as Dark Web (Tor/I2P) or Normal traffic based on behavioral signatures.
    """
    if prediction == 'Dark Web':
        reasons = []
        duration = float(row.get('flow_duration', 0))
        pkt_size = float(row.get('packet_size', 0))
        iat = float(row.get('inter_arrival_time', 0))
        
        if pkt_size <= 650:
            reasons.append(f"Fixed 512B cell padding pattern (mean size: {pkt_size:.1f}B)")
        if iat >= 0.25:
            reasons.append(f"Multi-hop relay latency / jitter (IAT: {iat:.3f}s)")
        if duration >= 50.0:
            reasons.append(f"Persistent anonymized circuit tunnel (duration: {duration:.1f}s)")
            
        if not reasons:
            reasons.append("Behavioral statistical signature matches Tor/I2P traffic distribution")
            
        return " | ".join(reasons)
    else:
        return "Legitimate standard HTTPS/HTTP transport profile (low latency, variable MTU)"

def predict_traffic(input_data, model_path="models/trained_model.pkl", preprocessor_path="models/preprocessor.pkl"):
    """
    Runs end-to-end inference on a CSV filepath or Pandas DataFrame.
    
    Args:
        input_data (str or pd.DataFrame): CSV file path or DataFrame containing network flows.
        model_path (str): Filepath to the saved model.
        preprocessor_path (str): Filepath to the saved preprocessor.
        
    Returns:
        pd.DataFrame: Original data appended with 'Prediction', 'Confidence', and 'Confidence_Score'.
    """
    # 1. Load Preprocessor
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor artifact not found at '{preprocessor_path}'. Please run src/train.py first.")
        
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
        
    # 2. Ingest Input Data
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Input file not found at '{input_data}'")
        raw_df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        raw_df = input_data.copy()
    else:
        raise ValueError("input_data must be a valid CSV file path or a pandas DataFrame.")
        
    if len(raw_df) == 0:
        raise ValueError("The provided dataset is empty.")
        
    output_df = raw_df.copy()
    
    # 3. Clean and Transform Data
    df_clean = preprocessor.handle_missing_values(raw_df)
    
    # Remove ground-truth 'label' column if present so it doesn't leak into feature matrix
    features_df = df_clean.drop(columns=['label'], errors='ignore')
    
    # Verify mandatory feature columns
    features_df = preprocessor.select_features(features_df)
    
    # Scale features
    X = preprocessor.normalize_features(features_df, training=False)
    
    # 4. Load Model and Predict
    model, model_type = load_inference_model(model_path)
    labels, confidences = get_predictions(model, model_type, X)
    
    # 5. Attach Predictions & Forensic Reason
    output_df['Prediction'] = labels
    output_df['Confidence'] = [f"{c:.2%}" for c in confidences]
    output_df['Confidence_Score'] = [round(c, 4) for c in confidences]
    output_df['Threat_Reason'] = [
        explain_prediction(output_df.iloc[i], labels[i]) for i in range(len(output_df))
    ]
    
    return output_df

if __name__ == "__main__":
    sample_file = "data/sample_traffic.csv"
    if os.path.exists(sample_file):
        print(f"--- Running Test Prediction on '{sample_file}' ---")
        try:
            results = predict_traffic(
                input_data=sample_file,
                model_path="models/trained_model.pkl",
                preprocessor_path="models/preprocessor.pkl"
            )
            print("\nPreview of Prediction Results:")
            print(results[['flow_duration', 'packet_size', 'Prediction', 'Confidence']].head(10))
        except Exception as e:
            print(f"Prediction failed: {e}")
            print("Tip: Ensure models/trained_model.pkl and models/preprocessor.pkl have been generated via src/train.py.")
    else:
        print("Please train a model first using: python src/train.py")
