import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure root dir is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import TrafficDataPreprocessor
from src.train import train_sklearn
from src.predict import predict_traffic, load_inference_model

@pytest.fixture
def sample_raw_data():
    """Provides a synthetic DataFrame for testing."""
    normal_df = pd.DataFrame({
        'flow_duration': [10.5, 3.2, 5.1, 15.0, 1.8, 8.4, 12.1, 4.5, 2.0, 6.7],
        'packet_size': [1420.0, 1480.0, 1390.0, 1450.0, 1500.0, 1410.0, 1440.0, 1430.0, 1490.0, 1380.0],
        'inter_arrival_time': [0.01, 0.008, 0.015, 0.02, 0.005, 0.012, 0.018, 0.009, 0.004, 0.011],
        'byte_count': [150000, 45000, 73000, 21000, 12000, 26000, 63000, 64000, 28000, 83000],
        'label': ['Normal'] * 10
    })
    dark_df = pd.DataFrame({
        'flow_duration': [120.3, 140.5, 98.6, 210.4, 115.3, 165.8, 190.1, 78.5, 128.4, 240.5],
        'packet_size': [480.0, 340.2, 490.1, 298.5, 420.0, 385.4, 310.0, 530.2, 460.8, 280.0],
        'inter_arrival_time': [0.65, 0.85, 0.52, 1.21, 0.61, 0.78, 0.92, 0.41, 0.65, 1.45],
        'byte_count': [45000, 47890, 48320, 62800, 48426, 63900, 58930, 41620, 59166, 67340],
        'label': ['Dark Web'] * 10
    })
    return pd.concat([normal_df, dark_df], ignore_index=True)

def test_preprocessor_clean_data(sample_raw_data):
    """Test standard preprocessing pipeline."""
    preprocessor = TrafficDataPreprocessor()
    df_clean = preprocessor.handle_missing_values(sample_raw_data)
    assert not df_clean.isnull().values.any(), "Cleaned data should not contain any nulls"
    
    scaled = preprocessor.normalize_features(df_clean, training=True)
    assert scaled.shape == (len(sample_raw_data), 4), f"Expected shape ({len(sample_raw_data)}, 4), got {scaled.shape}"
    assert preprocessor.is_fitted is True, "Preprocessor should be marked as fitted"

def test_preprocessor_handle_missing_values():
    """Test handling of NaN, null strings, and missing rows."""
    preprocessor = TrafficDataPreprocessor()
    dirty_data = pd.DataFrame({
        'flow_duration': [10.0, np.nan, 'invalid', None],
        'packet_size': [1400.0, '?', 1300.0, 1200.0],
        'inter_arrival_time': [0.01, 0.02, 0.03, 'NA'],
        'byte_count': [10000, 20000, np.nan, 40000]
    })
    
    clean_df = preprocessor.handle_missing_values(dirty_data)
    for col in preprocessor.feature_columns:
        assert not clean_df[col].isnull().any(), f"Column {col} still has NaNs after imputation"
        assert pd.api.types.is_numeric_dtype(clean_df[col]), f"Column {col} must be numeric"

def test_preprocessor_missing_required_columns():
    """Test error raising when mandatory feature column is absent."""
    preprocessor = TrafficDataPreprocessor()
    invalid_data = pd.DataFrame({
        'unrelated_col1': [1, 2],
        'unrelated_col2': [3, 4]
    })
    with pytest.raises(ValueError, match="missing required feature columns"):
        preprocessor.select_features(invalid_data)

def test_label_encoding():
    """Test label encoding and inverse decoding."""
    preprocessor = TrafficDataPreprocessor()
    labels = ['Normal', 'Dark Web', 'Normal', 'darkweb']
    encoded = preprocessor.encode_labels(labels)
    assert list(encoded) == [0, 1, 0, 1]
    
    decoded = preprocessor.decode_labels(encoded)
    assert decoded == ['Normal', 'Dark Web', 'Normal', 'Dark Web']

def test_training_and_artifacts(tmp_path):
    """Test training pipeline saves both model and preprocessor."""
    temp_dir = str(tmp_path)
    data_file = os.path.join(temp_dir, "sample.csv")
    
    # Train model
    model, preprocessor = train_sklearn(
        data_path=data_file,
        model_type="sklearn",
        output_dir=temp_dir
    )
    
    assert os.path.exists(os.path.join(temp_dir, "trained_model.pkl"))
    assert os.path.exists(os.path.join(temp_dir, "preprocessor.pkl"))

def test_prediction_pipeline(sample_raw_data, tmp_path):
    """Test end-to-end prediction execution."""
    temp_dir = str(tmp_path)
    data_file = os.path.join(temp_dir, "sample.csv")
    sample_raw_data.to_csv(data_file, index=False)
    
    # Train
    train_sklearn(data_path=data_file, model_type="sklearn", output_dir=temp_dir)
    
    # Predict
    model_path = os.path.join(temp_dir, "trained_model.pkl")
    preproc_path = os.path.join(temp_dir, "preprocessor.pkl")
    
    results = predict_traffic(
        input_data=sample_raw_data,
        model_path=model_path,
        preprocessor_path=preproc_path
    )
    
    assert 'Prediction' in results.columns
    assert 'Confidence' in results.columns
    assert 'Confidence_Score' in results.columns
    assert len(results) == len(sample_raw_data)
    assert set(results['Prediction'].unique()).issubset({'Normal', 'Dark Web'})
    assert all(0.5 <= score <= 1.0 for score in results['Confidence_Score'])
