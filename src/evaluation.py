import pandas as pd
import yaml
import os
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mlflow

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def evaluate_model(data_path, model_path, params):
    """Evaluate the trained model"""
    
    # Load model
    model = joblib.load(model_path)
    
    # Load processed data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Split data (use same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state']
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_names = X.columns.tolist()
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    print("\n=== Feature Importance ===")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Save evaluation results
    evaluation_results = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": feature_importance
    }
    
    eval_path = os.path.join(os.path.dirname(model_path), 'evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(f"\nEvaluation results saved to {eval_path}")

if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Load parameters
    params = load_params()
    
    # Evaluate model
    evaluate_model(
        params['data']['processed_path'],
        'model/model.pkl',
        params
    )