import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TrafficDataPreprocessor:
    """
    Preprocessor for Network Traffic Data.
    Handles missing values, feature selection, normalization, and label encoding.
    """
    DEFAULT_FEATURES = ['flow_duration', 'packet_size', 'inter_arrival_time', 'byte_count']
    LABEL_COLUMN = 'label'
    
    LABEL_MAPPING = {
        'Normal': 0,
        'normal': 0,
        0: 0,
        'Dark Web': 1,
        'dark web': 1,
        'DarkWeb': 1,
        'darkweb': 1,
        1: 1
    }
    
    INVERSE_MAPPING = {
        0: 'Normal',
        1: 'Dark Web'
    }

    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns or self.DEFAULT_FEATURES
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.numeric_medians = {}

    def handle_missing_values(self, df):
        """
        Cleans and imputes missing or malformed values.
        """
        df = df.copy()
        
        # Replace common string placeholders with NaN
        df.replace(['?', 'NA', 'N/A', 'null', 'None', '', ' '], np.nan, inplace=True)
        
        # Coerce feature columns to numeric
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Compute median or use cached median
                if not self.is_fitted:
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    self.numeric_medians[col] = float(median_val)
                
                # Fill missing with median
                fill_val = self.numeric_medians.get(col, 0.0)
                df[col] = df[col].fillna(fill_val)
                
        return df

    def select_features(self, df):
        """
        Validates and extracts required features.
        Raises ValueError if mandatory features are completely missing.
        """
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Input data is missing required feature columns: {missing_features}")
            
        return df[self.feature_columns].copy()

    def normalize_features(self, df, training=False):
        """
        Normalizes numeric features using StandardScaler.
        """
        df_feat = self.select_features(df)
        
        if training:
            scaled_array = self.scaler.fit_transform(df_feat)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                # If scaler hasn't been fitted yet, fit on the incoming data as fallback
                scaled_array = self.scaler.fit_transform(df_feat)
                self.is_fitted = True
            else:
                scaled_array = self.scaler.transform(df_feat)
                
        return scaled_array

    def encode_labels(self, labels):
        """
        Encodes categorical labels ('Normal', 'Dark Web') to integers (0, 1).
        """
        encoded = []
        for val in labels:
            if val in self.LABEL_MAPPING:
                encoded.append(self.LABEL_MAPPING[val])
            elif str(val).strip() in self.LABEL_MAPPING:
                encoded.append(self.LABEL_MAPPING[str(val).strip()])
            else:
                raise ValueError(f"Unknown traffic label '{val}'. Expected 'Normal' or 'Dark Web'.")
        return np.array(encoded, dtype=int)

    def decode_labels(self, numeric_predictions):
        """
        Decodes numeric predictions (0, 1) back to class strings.
        """
        return [self.INVERSE_MAPPING.get(int(p), 'Unknown') for p in numeric_predictions]

    def save(self, filepath):
        """Saves the preprocessor state to a pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Loads a preprocessor state from a pickle file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor artifact not found at: {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
