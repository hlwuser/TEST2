import pandas as pd
import yaml
import os
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def train_model(data_path, model_path, params):
    """Train the Random Forest model"""
    
    # Initialize DagsHub integration (optional)
    # dagshub.init(repo_owner='YOUR_USERNAME', repo_name='YOUR_REPO', mlflow=True)
    
    # Load processed data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state']
    )
    
    # Initialize MLflow
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(
            n_estimators=params['model']['n_estimators'],
            max_depth=params['model']['max_depth'],
            min_samples_split=params['model']['min_samples_split'],
            min_samples_leaf=params['model']['min_samples_leaf'],
            random_state=params['model']['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log parameters
        mlflow.log_params(params['model'])
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # Save metrics to JSON
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        metrics_path = os.path.join(os.path.dirname(model_path), 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Load parameters
    params = load_params()
    
    # Train model
    train_model(
        params['data']['processed_path'],
        'model/model.pkl',
        params
    )