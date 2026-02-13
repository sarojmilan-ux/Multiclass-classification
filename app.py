"""
Mushroom Classification Streamlit App
Deployed ML models for mushroom classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, matthews_corrcoef,
                            confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6347;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #1e1e1e;
    }
    .metric-card h3 {
        color: #1e1e1e;
    }
    .metric-card p {
        color: #2e2e2e;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🍄 Mushroom Classification App</p>', unsafe_allow_html=True)
st.markdown("### Predict whether a mushroom is Edible or Poisonous")

# Sidebar
st.sidebar.header("📊 Model Selection & Data Upload")
st.sidebar.markdown("---")

# Model selection
model_options = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "K-Nearest Neighbors": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "🤖 Select ML Model",
    options=list(model_options.keys())
)

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Test Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with mushroom features"
)

# In app.py, this code works in deployment:
if os.path.exists("model/test_data.csv"):
    test_df = pd.read_csv("model/test_data.csv")
    csv_data = test_df.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        label="📥 Download Sample Test Data",
        data=csv_data,
        file_name="mushroom_test_data.csv",
        mime="text/csv"
    )


st.sidebar.markdown("---")
st.sidebar.info("""
**Instructions:**
1. Select a model from the dropdown
2. Upload your test data CSV
3. View predictions and metrics
""")

def load_model(model_name):
    """Load the selected model"""
    model_path = f"model/{model_options[model_name]}"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"❌ Model file not found: {model_path}")
        return None

def encode_data(df, has_target=True):
    """Encode categorical features to numeric values"""
    df_encoded = df.copy()
    
    # Determine which columns to encode
    if has_target and 'class' in df_encoded.columns:
        # Encode target variable
        le_target = LabelEncoder()
        df_encoded['class'] = le_target.fit_transform(df_encoded['class'].astype(str))
        feature_cols = [col for col in df_encoded.columns if col != 'class']
    else:
        feature_cols = df_encoded.columns
    
    # Encode all feature columns
    for col in feature_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix using plotly"""
    labels = ['Poisonous', 'Edible']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        width=500
    )
    
    return fig

def plot_metrics_bar(metrics):
    """Plot metrics as bar chart"""
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Score': list(metrics.values())
    })
    
    fig = px.bar(
        metrics_df,
        x='Metric',
        y='Score',
        title='Model Performance Metrics',
        text='Score',
        color='Score',
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, 1.1],
        height=400,
        showlegend=False
    )
    
    return fig

# Main app logic
if uploaded_file is not None:
    try:
        # Load data
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! Shape: {df_raw.shape}")
        
        # Display data preview
        with st.expander("📄 View Data Preview (Raw)"):
            st.dataframe(df_raw.head(10))
        
        # Encode the data
        df = encode_data(df_raw, has_target='class' in df_raw.columns)
        
        # Check if 'class' column exists
        if 'class' in df.columns:
            X_test = df.drop('class', axis=1)
            y_test = df['class']
            has_labels = True
        else:
            X_test = df
            y_test = None
            has_labels = False
            st.warning("⚠️ No 'class' column found. Predictions will be made without evaluation metrics.")
        
        # Load model
        model = load_model(selected_model_name)
        
        if model is not None:
            st.markdown('<p class="sub-header">🔮 Making Predictions...</p>', unsafe_allow_html=True)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X_test)
            else:
                pred_proba = None
            
            # Display predictions
            pred_df = pd.DataFrame({
                'Sample': range(1, len(predictions) + 1),
                'Prediction': ['Edible' if p == 1 else 'Poisonous' for p in predictions]
            })
            
            if pred_proba is not None:
                pred_df['Confidence'] = np.max(pred_proba, axis=1)
            
            with st.expander("🔍 View All Predictions"):
                st.dataframe(pred_df)
            
            # Prediction distribution
            st.markdown('<p class="sub-header">📊 Prediction Distribution</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                pred_counts = pd.Series(predictions).value_counts()
                edible_count = pred_counts.get(1, 0)
                poisonous_count = pred_counts.get(0, 0)
                
                st.metric("🟢 Edible", edible_count)
                st.metric("🔴 Poisonous", poisonous_count)
            
            with col2:
                fig_pie = px.pie(
                    values=[poisonous_count, edible_count],
                    names=['Poisonous', 'Edible'],
                    title='Prediction Distribution',
                    color_discrete_sequence=['#FF6347', '#32CD32']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # If we have true labels, calculate metrics
            if has_labels and y_test is not None:
                st.markdown('<p class="sub-header">📈 Model Evaluation Metrics</p>', unsafe_allow_html=True)
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, predictions, pred_proba)
                
                # Display metrics in columns
                cols = st.columns(6)
                for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                    with cols[idx]:
                        st.metric(metric_name, f"{metric_value:.4f}")
                
                # Plot metrics
                st.plotly_chart(plot_metrics_bar(metrics), use_container_width=True)
                
                # Confusion Matrix
                st.markdown('<p class="sub-header">🎯 Confusion Matrix</p>', unsafe_allow_html=True)
                
                cm = confusion_matrix(y_test, predictions)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)
                
                with col2:
                    st.markdown("### Classification Report")
                    report = classification_report(
                        y_test,
                        predictions,
                        target_names=['Poisonous', 'Edible'],
                        output_dict=True
                    )
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"))
                
                # Additional insights
                st.markdown('<p class="sub-header">💡 Model Insights</p>', unsafe_allow_html=True)
                
                accuracy = metrics['Accuracy']
                if accuracy >= 0.95:
                    performance = "Excellent"
                    color = "green"
                elif accuracy >= 0.85:
                    performance = "Good"
                    color = "blue"
                elif accuracy >= 0.75:
                    performance = "Fair"
                    color = "orange"
                else:
                    performance = "Poor"
                    color = "red"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Overall Performance: <span style="color:{color}">{performance}</span></h3>
                    <p>The <b>{selected_model_name}</b> model achieved an accuracy of <b>{accuracy:.4f}</b> on the test dataset.</p>
                    <p>✓ True Positives (Correctly predicted Edible): <b>{cm[1][1]}</b></p>
                    <p>✓ True Negatives (Correctly predicted Poisonous): <b>{cm[0][0]}</b></p>
                    <p>✗ False Positives (Poisonous predicted as Edible): <b>{cm[0][1]}</b></p>
                    <p>✗ False Negatives (Edible predicted as Poisonous): <b>{cm[1][0]}</b></p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    # Landing page with instructions
    st.markdown("""
    ## 👋 Welcome to the Mushroom Classification App!
    
    This application uses machine learning to predict whether a mushroom is **edible** or **poisonous** based on its characteristics.
    
    ### 🚀 How to Use:
    
    1. **Select a Model**: Choose from 6 different ML models in the sidebar
    2. **Upload Data**: Upload your test dataset (CSV format)
    3. **View Results**: See predictions, metrics, and confusion matrix
    
    ### 📊 Available Models:
    
    - **Logistic Regression**: Linear classification model
    - **Decision Tree**: Tree-based classification
    - **K-Nearest Neighbors**: Instance-based learning
    - **Naive Bayes**: Probabilistic classifier
    - **Random Forest**: Ensemble of decision trees
    - **XGBoost**: Gradient boosting ensemble
    
    ### 📁 Data Format:
    
    Your CSV file should contain mushroom features as columns. If it includes a 'class' column:
    - `0` = Poisonous
    - `1` = Edible
    
    ---
    
    **Note**: This app uses models trained on the UCI Mushroom Dataset. For demonstration purposes, upload the test data provided.
    """)
    
    # Display sample data format
    st.markdown("### 📋 Sample Data Format")
    sample_data = {
        'cap-shape': [5, 2, 0],
        'cap-surface': [3, 3, 3],
        'cap-color': [4, 9, 9],
        'bruises': [1, 1, 1],
        'odor': [6, 0, 0],
        'class': [0, 1, 1]
    }
    st.dataframe(pd.DataFrame(sample_data))
    st.caption("Sample showing encoded categorical features")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🍄 Mushroom Classification App | Built with Streamlit</p>
    <p>Dataset: <a href='https://archive.ics.uci.edu/dataset/73/mushroom'>UCI Mushroom Dataset</a></p>
</div>
""", unsafe_allow_html=True)
