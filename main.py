import numpy as np
import pandas as pd
import torch
from preprocess import normalize_features, create_sequences
from model import OrbitalLSTM
from train import train_orbital_model
from evaluate import detect_close_approaches

def run_demo():
    print("🚀 Starting OrbitGuard End-to-End Demo...\n")
    
    # 1. Generate Synthetic Trajectory Data (Simulating thousands of km)
    print("[1/4] Generating synthetic orbital data...")
    time_steps = 1000
    
    # Simulate Satellite A (Sine/Cosine waves to simulate orbit)
    sat_a_x = np.sin(np.linspace(0, 20, time_steps)) * 6000
    sat_a_y = np.cos(np.linspace(0, 20, time_steps)) * 6000
    sat_a_z = np.zeros(time_steps)
    df_a = pd.DataFrame({'sat_id': 'SAT_A', 'x': sat_a_x, 'y': sat_a_y, 'z': sat_a_z})
    
    # Simulate Satellite B (Slightly offset orbit)
    sat_b_x = np.cos(np.linspace(0, 20, time_steps)) * 6000
    sat_b_y = np.sin(np.linspace(0, 20, time_steps)) * 6000
    sat_b_z = np.ones(time_steps) * 15  # 15km Z-axis offset
    df_b = pd.DataFrame({'sat_id': 'SAT_B', 'x': sat_b_x, 'y': sat_b_y, 'z': sat_b_z})
    
    df_raw = pd.concat([df_a, df_b]).reset_index(drop=True)
    
    # 2. Preprocess Data
    print("[2/4] Normalizing features and creating sequential rolling windows...")
    df_norm, scaler = normalize_features(df_raw, feature_cols=['x', 'y', 'z'])
    
    X, y = create_sequences(
        df_norm, 
        feature_cols=['x', 'y', 'z'], 
        target_cols=['x', 'y', 'z'], 
        window_size=10, 
        horizon=1, 
        group_col='sat_id'
    )
    
    # Split into train/validation sets (80/20)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # 3. Train the Model
    print("\n[3/4] 🧠 Initializing and Training OrbitalLSTM (5 Epochs)...")
    model = OrbitalLSTM(input_size=3, hidden_size=32, num_layers=2, output_size=3)
    
    # Run the training loop (defined in train.py)
    train_orbital_model(
        model, X_train, y_train, X_val, y_val, 
        epochs=5, batch_size=64
    )
    
    # 4. Evaluate and Check for Collisions
    print("\n[4/4] 📊 Simulating live inference and collision detection...")
    model.eval()
    with torch.no_grad():
        # Predict the next coordinate for SAT_A and SAT_B using the first two validation windows
        dummy_pred_norm = model(torch.tensor(X_val[:2], dtype=torch.float32)).numpy()
        
    # Un-normalize back to kilometers
    predicted_km = scaler.inverse_transform(dummy_pred_norm)
    
    positions = {
        'SAT_A': predicted_km[0],
        'SAT_B': predicted_km[1]
    }
    
    print("\nProjected Coordinates (km):")
    print(f"SAT_A: {positions['SAT_A']}")
    print(f"SAT_B: {positions['SAT_B']}")
    
    # Run collision check threshold at 50 km
    alerts = detect_close_approaches(positions, threshold_km=50.0)
    if not alerts:
        print("\n✅ All clear! No collision risks detected in the current projection.")
        
    print("\n🏁 Demo Complete!")

if __name__ == "__main__":
    run_demo()
