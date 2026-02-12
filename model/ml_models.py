"""
Machine Learning Classification Models Implementation
Assignment 2 - Heart Disease Prediction

This module implements 6 different classification models:
1. Logistic Regression
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

All models are evaluated using 6 metrics:
- Accuracy, AUC Score, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
import pickle
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseClassifier:
    """
    A comprehensive ML classifier for Heart Disease prediction
    Implements and compares 6 different classification algorithms
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_dataset(self):
        """
        Load and create the Heart Disease dataset
        Features: 13 numeric features
        Target: Binary classification (0: No disease, 1: Disease)
        """
        np.random.seed(self.random_state)
        
        # Generate realistic heart disease dataset with 13 features
        n_samples = 1000
        
        # Feature generation with realistic distributions
        data = {
            'age': np.random.normal(54, 12, n_samples).clip(20, 80),
            'sex': np.random.binomial(1, 0.68, n_samples),  # More males in heart disease studies
            'cp': np.random.randint(0, 4, n_samples),  # Chest pain type
            'trestbps': np.random.normal(132, 18, n_samples).clip(90, 200),  # Resting blood pressure
            'chol': np.random.normal(247, 52, n_samples).clip(120, 400),  # Cholesterol
            'fbs': np.random.binomial(1, 0.15, n_samples),  # Fasting blood sugar > 120
            'restecg': np.random.randint(0, 3, n_samples),  # Resting ECG results
            'thalach': np.random.normal(150, 25, n_samples).clip(70, 200),  # Max heart rate
            'exang': np.random.binomial(1, 0.33, n_samples),  # Exercise induced angina
            'oldpeak': np.random.exponential(1.0, n_samples).clip(0, 6.2),  # ST depression
            'slope': np.random.randint(0, 3, n_samples),  # Slope of peak exercise ST segment
            'ca': np.random.poisson(0.7, n_samples).clip(0, 3),  # Number of major vessels
            'thal': np.random.randint(0, 4, n_samples)  # Thalassemia
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic target based on medical knowledge
        # Higher risk factors: older age, male, high cholesterol, chest pain, etc.
        risk_score = (
            (df['age'] > 55).astype(int) * 2 +
            df['sex'] * 1.5 +
            (df['cp'] > 0).astype(int) * 2 +
            (df['trestbps'] > 140).astype(int) * 1 +
            (df['chol'] > 240).astype(int) * 2 +
            df['fbs'] * 1 +
            (df['thalach'] < 150).astype(int) * 1.5 +
            df['exang'] * 2 +
            (df['oldpeak'] > 1).astype(int) * 1.5 +
            df['ca'] * 0.5 +
            (df['thal'] > 0).astype(int) * 1
        )
        
        # Convert risk score to probability and then binary outcome
        prob = 1 / (1 + np.exp(-(risk_score - 6)))
        df['target'] = np.random.binomial(1, prob, n_samples)
        
        self.df = df
        print(f"Dataset created successfully!")
        print(f"Shape: {df.shape}")
        print(f"Features: {df.columns.tolist()[:-1]}")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def prepare_data(self, test_size=0.2):
        """Prepare and split the dataset"""
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
    def initialize_models(self):
        """Initialize all 6 ML models"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_estimators=100
            )
        }
        print("All models initialized successfully!")
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all 6 evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1': f1_score(y_true, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        return metrics
    
    def train_and_evaluate(self):
        """Train all models and evaluate their performance"""
        self.results = {}
        
        print("\n" + "="*60)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # Train the model
            if name in ['Logistic Regression', 'KNN', 'Naive Bayes']:
                # These models benefit from scaled features
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)
            else:
                # Tree-based models can work with original features
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'metrics': metrics
            }
            
            # Print results
            print(f"✅ {name} - Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*60)
    
    def display_results_table(self):
        """Display comprehensive results table"""
        results_data = []
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            results_data.append([
                model_name,
                f"{metrics['Accuracy']:.4f}",
                f"{metrics['AUC']:.4f}",
                f"{metrics['Precision']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['F1']:.4f}",
                f"{metrics['MCC']:.4f}"
            ])
        
        results_df = pd.DataFrame(results_data, columns=[
            'ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'
        ])
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON TABLE")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        return results_df
    
    def generate_model_observations(self):
        """Generate observations about each model's performance"""
        observations = {}
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            
            if model_name == 'Logistic Regression':
                obs = f"Shows balanced performance with Accuracy: {metrics['Accuracy']:.3f} and AUC: {metrics['AUC']:.3f}. "
                obs += "Good interpretability and handles linear relationships well. "
                if metrics['AUC'] > 0.85:
                    obs += "Excellent discrimination capability."
                else:
                    obs += "Moderate discrimination capability."
            
            elif model_name == 'Decision Tree':
                obs = f"Achieves Accuracy: {metrics['Accuracy']:.3f} with high interpretability. "
                if metrics['Accuracy'] < 0.80:
                    obs += "May be overfitting to training data. "
                obs += f"F1-score of {metrics['F1']:.3f} indicates balanced precision-recall trade-off."
            
            elif model_name == 'KNN':
                obs = f"Non-parametric approach with Accuracy: {metrics['Accuracy']:.3f}. "
                obs += "Performance depends on local neighborhood patterns. "
                if metrics['MCC'] > 0.6:
                    obs += "Good correlation between predictions and actual values."
                else:
                    obs += "Moderate correlation, may need parameter tuning."
            
            elif model_name == 'Naive Bayes':
                obs = f"Probabilistic classifier with Accuracy: {metrics['Accuracy']:.3f}. "
                obs += "Assumes feature independence which may limit performance. "
                if metrics['Precision'] > 0.80:
                    obs += "Low false positive rate, good for screening applications."
            
            elif model_name == 'Random Forest':
                obs = f"Ensemble method achieving Accuracy: {metrics['Accuracy']:.3f}. "
                obs += "Reduces overfitting compared to single decision tree. "
                if metrics['AUC'] > 0.90:
                    obs += "Excellent performance with strong generalization capability."
                else:
                    obs += "Good performance with decent generalization."
            
            elif model_name == 'XGBoost':
                obs = f"Advanced ensemble with Accuracy: {metrics['Accuracy']:.3f}. "
                obs += "Gradient boosting provides strong predictive performance. "
                if metrics['AUC'] > 0.90:
                    obs += "Top-tier performance suitable for production deployment."
                else:
                    obs += "Strong performance with good robustness."
            
            observations[model_name] = obs
        
        return observations
    
    def plot_model_comparison(self):
        """Create visualization comparing all models"""
        metrics_data = []
        model_names = []
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            metrics_data.append([
                metrics['Accuracy'], metrics['AUC'], metrics['Precision'],
                metrics['Recall'], metrics['F1'], metrics['MCC']
            ])
            model_names.append(model_name)
        
        metrics_df = pd.DataFrame(metrics_data, 
                                columns=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                                index=model_names)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML Models Performance Comparison - Heart Disease Prediction', fontsize=16, fontweight='bold')
        
        metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, metric in enumerate(metrics_list):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            bars = ax.bar(model_names, metrics_df[metric], color=colors[i], alpha=0.8)
            ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=12)
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/bajaja/ai_ml_work/ml_2/ml_2/assignment_solution/model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save all trained models"""
        import os
        model_dir = '/home/bajaja/ai_ml_work/ml_2/ml_2/assignment_solution/model'
        
        for name, results in self.results.items():
            model = results['model']
            filename = f"{name.lower().replace(' ', '_')}_model.pkl"
            filepath = os.path.join(model_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n✅ All models saved in {model_dir}")

def main():
    """Main execution function"""
    print("🤖 HEART DISEASE PREDICTION - ML CLASSIFICATION MODELS")
    print("="*60)
    print("Implementing 6 ML models with comprehensive evaluation")
    print("="*60)
    
    # Initialize classifier
    classifier = HeartDiseaseClassifier(random_state=42)
    
    # Load and prepare data
    print("\n📊 STEP 1: Loading Dataset...")
    classifier.load_dataset()
    
    print("\n🔧 STEP 2: Preparing Data...")
    classifier.prepare_data()
    
    print("\n🤖 STEP 3: Initializing Models...")
    classifier.initialize_models()
    
    print("\n🎯 STEP 4: Training and Evaluation...")
    classifier.train_and_evaluate()
    
    print("\n📋 STEP 5: Results Summary...")
    results_df = classifier.display_results_table()
    
    print("\n💭 STEP 6: Model Observations...")
    observations = classifier.generate_model_observations()
    
    print("\nMODEL PERFORMANCE OBSERVATIONS:")
    print("="*50)
    for model_name, observation in observations.items():
        print(f"\n{model_name}:")
        print(f"  {observation}")
    
    print("\n📊 STEP 7: Creating Visualizations...")
    classifier.plot_model_comparison()
    
    print("\n💾 STEP 8: Saving Models...")
    classifier.save_models()
    
    print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return classifier, results_df, observations

if __name__ == "__main__":
    classifier, results_df, observations = main()