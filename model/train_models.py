"""
Mushroom Classification - Model Training Script
This script trains 6 ML models and calculates all evaluation metrics
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import joblib

def load_and_preprocess_data():
    """Load mushroom dataset and preprocess"""
    print("Loading mushroom dataset...")
    
    # Load data from CSV file
    df = pd.read_csv('mushrooms.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)-1}")
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum().sum()}")
    
    # Encode target variable (e=edible=1, p=poisonous=0)
    le_target = LabelEncoder()
    df['class'] = le_target.fit_transform(df['class'])
    
    # Encode all categorical features
    le_dict = {}
    for col in df.columns[1:]:  # Skip target column
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Split features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    print(f"\nClass distribution:")
    print(f"Edible (1): {sum(y==1)} ({sum(y==1)/len(y)*100:.2f}%)")
    print(f"Poisonous (0): {sum(y==0)} ({sum(y==0)/len(y)*100:.2f}%)")
    
    return X, y, le_target

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all required evaluation metrics"""
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # AUC score requires probability predictions
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['auc'] = 0.0
    
    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and calculate metrics"""
    
    results = {}
    models_dict = {}
    
    # 1. Logistic Regression
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)
    results['Logistic Regression'] = calculate_metrics(y_test, lr_pred, lr_proba)
    models_dict['Logistic Regression'] = lr_model
    print("✓ Logistic Regression completed")
    
    # 2. Decision Tree Classifier
    print("\n" + "="*60)
    print("Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_proba = dt_model.predict_proba(X_test)
    results['Decision Tree'] = calculate_metrics(y_test, dt_pred, dt_proba)
    models_dict['Decision Tree'] = dt_model
    print("✓ Decision Tree completed")
    
    # 3. K-Nearest Neighbors
    print("\n" + "="*60)
    print("Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_proba = knn_model.predict_proba(X_test)
    results['KNN'] = calculate_metrics(y_test, knn_pred, knn_proba)
    models_dict['KNN'] = knn_model
    print("✓ KNN completed")
    
    # 4. Naive Bayes (Multinomial)
    print("\n" + "="*60)
    print("Training Naive Bayes (Multinomial)...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_proba = nb_model.predict_proba(X_test)
    results['Naive Bayes'] = calculate_metrics(y_test, nb_pred, nb_proba)
    models_dict['Naive Bayes'] = nb_model
    print("✓ Naive Bayes completed")
    
    # 5. Random Forest (Ensemble)
    print("\n" + "="*60)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    results['Random Forest'] = calculate_metrics(y_test, rf_pred, rf_proba)
    models_dict['Random Forest'] = rf_model
    print("✓ Random Forest completed")
    
    # 6. XGBoost (Ensemble)
    print("\n" + "="*60)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, max_depth=6, 
                              use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    results['XGBoost'] = calculate_metrics(y_test, xgb_pred, xgb_proba)
    models_dict['XGBoost'] = xgb_model
    print("✓ XGBoost completed")
    
    return results, models_dict

def print_results_table(results):
    """Print results in a formatted table"""
    print("\n" + "="*100)
    print("MODEL EVALUATION RESULTS")
    print("="*100)
    
    # Create DataFrame for better visualization
    df_results = pd.DataFrame(results).T
    df_results = df_results[['accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc']]
    df_results.columns = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    print("\n" + df_results.to_string())
    print("\n" + "="*100)
    
    return df_results

def save_models(models_dict):
    """Save trained models"""
    print("\nSaving models...")
    for name, model in models_dict.items():
        filename = f"model/{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, filename)
        print(f"✓ Saved {name} to {filename}")

def main():
    """Main execution function"""
    print("="*60)
    print("MUSHROOM CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess data
    X, y, le_target = load_and_preprocess_data()
    
    # Split data
    print("\nSplitting data into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train and evaluate all models
    results, models_dict = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Print results table
    df_results = print_results_table(results)
    
    # Save results to CSV
    df_results.to_csv('model/model_results.csv')
    print("\n✓ Results saved to model/model_results.csv")
    
    # Save all models
    save_models(models_dict)
    
    # Save test data for Streamlit app
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('model/test_data.csv', index=False)
    print("✓ Test data saved to model/test_data.csv")
    
    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
