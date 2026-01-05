import pandas as pd
import yaml
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def preprocess_data(input_path, output_path):
    """Preprocess the raw data"""
    # Load data
    df = pd.read_csv(input_path)
    
    # Create label encoders for categorical columns
    label_encoders = {}
    categorical_columns = ['gender', 'occupation', 'education_level', 'marital_status', 'loan_status']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Load parameters
    params = load_params()
    
    # Preprocess data
    preprocess_data(
        params['data']['raw_path'],
        params['data']['processed_path']
    )