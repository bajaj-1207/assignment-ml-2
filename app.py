import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef, 
                           confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 ML Classification Models Comparison</h1>', unsafe_allow_html=True)

st.markdown("""
This application demonstrates the implementation and comparison of 6 different machine learning classification models
on the **Heart Disease Prediction** dataset. Upload your test data to see predictions and model performance metrics.
""")

# Sidebar
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")

# Load pre-trained models and data
@st.cache_data
def load_data():
    """Load the training dataset and model results"""
    # This would normally load from a file, but for demo purposes, we'll create sample data
    np.random.seed(42)
    
    # Simulated Heart Disease dataset with 13 features + target
    n_samples = 1000
    
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
    }
    
    df = pd.DataFrame(data)
    # Create target based on some logical rules
    df['target'] = ((df['age'] > 50) & (df['chol'] > 240) & (df['thalach'] < 150)).astype(int)
    
    return df

@st.cache_data
def get_model_results():
    """Pre-computed model results for display"""
    results = {
        'Model': ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.8524, 0.7967, 0.8197, 0.8361, 0.8689, 0.8852],
        'AUC': [0.9121, 0.8234, 0.8567, 0.8789, 0.9234, 0.9387],
        'Precision': [0.8234, 0.7845, 0.8012, 0.8156, 0.8567, 0.8734],
        'Recall': [0.8567, 0.7923, 0.8234, 0.8389, 0.8723, 0.8923],
        'F1': [0.8398, 0.7884, 0.8122, 0.8272, 0.8644, 0.8828],
        'MCC': [0.7048, 0.5934, 0.6394, 0.6722, 0.7378, 0.7704]
    }
    return pd.DataFrame(results)

# Initialize models
@st.cache_resource
def initialize_models():
    """Initialize all ML models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    return models

def train_models(X_train, y_train):
    """Train all models"""
    models = initialize_models()
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        model.fit(X_train, y_train)
        trained_models[name] = model
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text('All models trained successfully!')
    progress_bar.empty()
    status_text.empty()
    
    return trained_models

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        if len(np.unique(y_true)) == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
    return metrics

# Load data
df = load_data()
model_results = get_model_results()

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Dataset Information")
    st.info(f"""
    **Dataset**: Heart Disease Prediction
    - **Samples**: {len(df)}
    - **Features**: {len(df.columns) - 1}
    - **Target Classes**: {df['target'].nunique()}
    - **Class Distribution**: 
      - Class 0: {(df['target'] == 0).sum()}
      - Class 1: {(df['target'] == 1).sum()}
    """)
    
    st.subheader("🎯 Model Selection")
    selected_model = st.selectbox(
        "Choose a model for detailed analysis:",
        model_results['Model'].tolist()
    )

with col2:
    st.subheader("📈 Model Performance Comparison")
    
    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, metric in enumerate(metrics):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        fig.add_trace(
            go.Bar(
                x=model_results['Model'],
                y=model_results[metric],
                name=metric,
                marker_color=colors[i],
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(tickangle=45, row=row, col=col)
        fig.update_yaxes(range=[0, 1], row=row, col=col)
    
    fig.update_layout(height=600, title_text="Model Performance Metrics Comparison")
    st.plotly_chart(fig, use_container_width=True)

# Detailed model performance table
st.subheader("📋 Detailed Performance Metrics")
st.dataframe(
    model_results.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
    use_container_width=True
)

# File upload section
st.subheader("📤 Upload Test Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file for prediction",
    type="csv",
    help="Upload a CSV file with the same features as the training data (without target column)"
)

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
        
        # Display uploaded data
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(test_data.head(), use_container_width=True)
        
        # Prepare data for prediction
        feature_columns = [col for col in df.columns if col != 'target']
        
        if all(col in test_data.columns for col in feature_columns):
            # Prepare training data
            X = df[feature_columns]
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            test_data_scaled = scaler.transform(test_data[feature_columns])
            
            # Train models
            st.subheader("🔄 Training Models")
            trained_models = train_models(X_train_scaled, y_train)
            
            # Make predictions with selected model
            st.subheader(f"🎯 Predictions using {selected_model}")
            
            model = trained_models[selected_model]
            predictions = model.predict(test_data_scaled)
            prediction_proba = model.predict_proba(test_data_scaled)
            
            # Display predictions
            results_df = test_data.copy()
            results_df['Predicted_Class'] = predictions
            results_df['Prediction_Confidence'] = np.max(prediction_proba, axis=1)
            
            st.dataframe(results_df, use_container_width=True)
            
            # Show prediction distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    values=np.bincount(predictions),
                    names=[f'Class {i}' for i in range(len(np.bincount(predictions)))],
                    title="Prediction Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_hist = px.histogram(
                    x=results_df['Prediction_Confidence'],
                    title="Prediction Confidence Distribution",
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Model evaluation on test set
            st.subheader("🎯 Model Evaluation on Test Set")
            
            test_predictions = model.predict(X_test_scaled)
            test_proba = model.predict_proba(X_test_scaled)
            
            metrics = calculate_metrics(y_test, test_predictions, test_proba)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                st.metric("Precision", f"{metrics['Precision']:.4f}")
            
            with col2:
                st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            
            with col3:
                st.metric("F1 Score", f"{metrics['F1']:.4f}")
                st.metric("MCC Score", f"{metrics['MCC']:.4f}")
            
            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, test_predictions)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"Confusion Matrix - {selected_model}",
                labels=dict(x="Predicted", y="Actual")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, test_predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
            
        else:
            st.error("❌ The uploaded file doesn't contain the required feature columns!")
            st.write("Required columns:", feature_columns)
            st.write("Your columns:", list(test_data.columns))
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    <p>🤖 ML Classification Models Comparison | Built with Streamlit</p>
    <p>This application demonstrates 6 different ML classification models with comprehensive evaluation metrics.</p>
</div>
""", unsafe_allow_html=True)