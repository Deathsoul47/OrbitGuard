import pytest
import numpy as np
import pandas as pd
import torch

from preprocess import impute_missing_values, normalize_features, create_sequences
from model import OrbitalLSTM

class TestPreprocess:
    """Test suite for data preprocessing module."""

    def test_impute_missing_values(self):
        # Create dataset with two distinct satellite trajectories
        df = pd.DataFrame({
            'sat_id': ['A', 'A', 'A', 'B', 'B', 'B'],
            'x': [1.0, np.nan, 3.0, 10.0, np.nan, 30.0]
        })
        
        # Test grouped interpolation
        df_imputed = impute_missing_values(df, strategy='interpolate', group_col='sat_id')
        
        # Sat A: midpoint between 1.0 and 3.0 is 2.0
        assert df_imputed['x'].iloc[1] == 2.0
        # Sat B: midpoint between 10.0 and 30.0 is 20.0
        assert df_imputed['x'].iloc[4] == 20.0

    def test_normalize_features(self):
        df = pd.DataFrame({
            'x': [0.0, 50.0, 100.0],
            'y': [-10.0, 0.0, 10.0]
        })
        
        df_norm, scaler = normalize_features(df, feature_cols=['x', 'y'])
        
        # StandardScaler uses population std deviation (ddof=0)
        assert np.isclose(df_norm['x'].mean(), 0.0)
        assert np.isclose(df_norm['x'].std(ddof=0), 1.0)
        assert np.isclose(df_norm['y'].mean(), 0.0)
        assert np.isclose(df_norm['y'].std(ddof=0), 1.0)

    def test_create_sequences(self):
        # 10 rows for a single satellite
        df = pd.DataFrame({
            'sat_id': ['A'] * 10,
            'x': np.arange(10),
            'y': np.arange(10, 20)
        })
        
        window_size = 3
        horizon = 1
        X, y = create_sequences(
            df, 
            feature_cols=['x', 'y'], 
            target_cols=['x', 'y'], 
            window_size=window_size, 
            horizon=horizon, 
            group_col='sat_id'
        )
        
        # Total rows = 10. 
        # Expected sequence count = total_rows - window_size - horizon + 1 = 10 - 3 - 1 + 1 = 7
        assert X.shape == (7, 3, 2)
        assert y.shape == (7, 2)
        
        # Verify chronological slicing of the first sequence
        assert np.array_equal(X[0], np.array([[0, 10], [1, 11], [2, 12]]))
        # Target should be the step immediately following the window
        assert np.array_equal(y[0], np.array([3, 13]))


class TestModel:
    """Test suite for the OrbitalLSTM network."""

    def test_orbital_lstm_tensor_shapes(self):
        batch_size = 16
        seq_len = 5
        input_size = 3      # e.g., (x, y, z)
        output_size = 3     # predicting future (x, y, z)
        hidden_size = 32
        
        model = OrbitalLSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=2, 
            output_size=output_size,
            dropout=0.1
        )
        
        # Simulate batch of historical trajectory windows
        dummy_input = torch.randn(batch_size, seq_len, input_size)
        
        # Execute forward pass
        predictions = model(dummy_input)
        
        # The output must exactly match (batch_size, output_size) to pair with target tensors
        assert predictions.shape == (batch_size, output_size)
        
        # Ensure gradients can flow (requires_grad is True)
        assert predictions.requires_grad
