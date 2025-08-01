"""
Logistic Regression modeling step for landslide susceptibility analysis.

This step implements Logistic Regression using scikit-learn, optimized for
binary classification tasks with comprehensive statistical analysis.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, log_loss
)
from scipy import stats
from typing import Dict, Any, Optional, Tuple
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class LogisticRegressionStep(BaseStep):
    """
    Logistic Regression training and analysis step.
    
    Specialized for landslide susceptibility mapping with statistical
    significance testing and coefficient interpretation.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[list] = None
        self.coefficients_: Optional[np.ndarray] = None
        
    def get_step_type(self) -> str:
        return "logistic_regression"
    
    def get_required_inputs(self) -> list:
        return ['training_data']
    
    def get_outputs(self) -> list:
        return ['trained_model', 'model_metrics', 'coefficient_analysis', 'statistical_tests']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate logistic regression hyperparameters."""
        # Validate solver
        solver = hyperparameters.get('solver', 'liblinear')
        valid_solvers = ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
        if solver not in valid_solvers:
            logger.error(f"Invalid solver: {solver}. Must be one of {valid_solvers}")
            return False
        
        # Validate penalty
        penalty = hyperparameters.get('penalty', 'l2')
        valid_penalties = ['l1', 'l2', 'elasticnet', 'none']
        if penalty not in valid_penalties:
            logger.error(f"Invalid penalty: {penalty}. Must be one of {valid_penalties}")
            return False
        
        # Validate penalty-solver combinations
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            logger.error("L1 penalty only supported with 'liblinear' and 'saga' solvers")
            return False
        
        if penalty == 'elasticnet' and solver != 'saga':
            logger.error("Elasticnet penalty only supported with 'saga' solver")
            return False
        
        # Validate numeric parameters
        numeric_params = {
            'C': (float, 1e-6, 1e6),
            'max_iter': (int, 100, 10000),
            'test_size': (float, 0.1, 0.5),
            'random_state': (int, 0, 999999)
        }
        
        for param, (param_type, min_val, max_val) in numeric_params.items():
            if param in hyperparameters:
                value = hyperparameters[param]
                if not isinstance(value, param_type):
                    logger.error(f"Parameter {param} must be of type {param_type}")
                    return False
                if not (min_val <= value <= max_val):
                    logger.error(f"Parameter {param} must be between {min_val} and {max_val}")
                    return False
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute Logistic Regression training and analysis.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing execution results and outputs
        """
        try:
            hyperparameters = context.get('hyperparameters', {})
            
            # Validate hyperparameters
            if not self.validate_hyperparameters(hyperparameters):
                return {
                    'status': 'failed',
                    'error': 'Invalid hyperparameters',
                    'outputs': {}
                }
            
            # Load training data
            training_data_path = context['inputs']['training_data']
            training_data = self._load_training_data(training_data_path)
            
            if training_data is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to load training data',
                    'outputs': {}
                }
            
            # Prepare features and target
            X, y, feature_names = self._prepare_features_target(
                training_data, hyperparameters
            )
            
            # Split data for training and testing
            test_size = hyperparameters.get('test_size', 0.2)
            random_state = hyperparameters.get('random_state', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features if requested
            if hyperparameters.get('scale_features', True):
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                self.scaler = None
            
            # Create and train model
            model = self._create_model(hyperparameters)
            
            logger.info(f"Training Logistic Regression with {X_train.shape[0]} samples, {X_train.shape[1]} features")
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Perform coefficient analysis
            coefficient_analysis = self._analyze_coefficients(
                model, feature_names, X_train_scaled, y_train
            )
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(
                model, X_train_scaled, y_train, feature_names
            )
            
            # Store model and feature information
            self.model = model
            self.feature_names_ = feature_names
            self.coefficients_ = model.coef_[0]
            
            # Save model if requested
            model_path = None
            if hyperparameters.get('save_model', True):
                model_path = self._save_model(model, context, hyperparameters)
            
            # Prepare outputs
            outputs = {
                'trained_model': {
                    'model_object': model,
                    'scaler_object': self.scaler,
                    'model_path': model_path,
                    'n_features': len(feature_names),
                    'intercept': float(model.intercept_[0])
                },
                'model_metrics': metrics,
                'coefficient_analysis': coefficient_analysis,
                'statistical_tests': statistical_tests
            }
            
            logger.info(f"Logistic Regression training completed successfully")
            logger.info(f"Model performance - AUC: {metrics['roc_auc']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
            
            return {
                'status': 'success',
                'message': 'Logistic Regression model trained successfully',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'training_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'n_features': X_train.shape[1]
                }
            }
            
        except Exception as e:
            logger.error(f"Logistic Regression training failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_training_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load training data from file."""
        try:
            if data_path.endswith('.csv'):
                return pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                return pd.read_parquet(data_path)
            elif data_path.endswith('.json'):
                return pd.read_json(data_path)
            else:
                logger.error(f"Unsupported file format: {data_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            return None
    
    def _prepare_features_target(self, data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prepare feature matrix and target vector from training data."""
        # Get target column
        target_column = hyperparameters.get('target_column', 'target')
        if target_column not in data.columns:
            # Try common target column names
            possible_targets = ['landslide', 'susceptibility', 'class', 'label', 'target']
            for col in possible_targets:
                if col in data.columns:
                    target_column = col
                    break
            else:
                raise ValueError(f"Target column not found. Specify 'target_column' in hyperparameters")
        
        # Exclude non-feature columns
        exclude_columns = hyperparameters.get('exclude_columns', [])
        exclude_columns.extend([target_column, 'geometry', 'id', 'sample_id'])
        
        # Select feature columns
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Handle missing values
        data_clean = data[feature_columns + [target_column]].dropna()
        
        X = data_clean[feature_columns].values
        y = data_clean[target_column].values.astype(int)
        
        # Ensure binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, found {len(unique_classes)} classes: {unique_classes}")
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y, feature_columns
    
    def _create_model(self, hyperparameters: Dict[str, Any]) -> LogisticRegression:
        """Create Logistic Regression model."""
        params = {
            'penalty': hyperparameters.get('penalty', 'l2'),
            'C': hyperparameters.get('C', 1.0),
            'solver': hyperparameters.get('solver', 'liblinear'),
            'max_iter': hyperparameters.get('max_iter', 1000),
            'random_state': hyperparameters.get('random_state', 42),
            'class_weight': hyperparameters.get('class_weight', 'balanced')
        }
        
        # Add l1_ratio for elasticnet penalty
        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = hyperparameters.get('l1_ratio', 0.5)
        
        return LogisticRegression(**params)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        logloss = log_loss(y_true, y_pred_proba)
        
        # Get ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        # Get Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'log_loss': float(logloss),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'precision': float(class_report['weighted avg']['precision']),
            'recall': float(class_report['weighted avg']['recall']),
            'f1_score': float(class_report['weighted avg']['f1-score']),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
    
    def _analyze_coefficients(self, model: LogisticRegression, feature_names: list,
                            X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze model coefficients and their significance."""
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Calculate odds ratios
        odds_ratios = np.exp(coefficients)
        
        # Create coefficient DataFrame
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'odds_ratio': odds_ratios,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Add coefficient confidence intervals (approximate)
        # Note: For exact confidence intervals, we'd need the covariance matrix
        std_errors = self._estimate_coefficient_std_errors(model, X, y)
        if std_errors is not None:
            z_scores = coefficients / std_errors
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            
            coef_df['std_error'] = std_errors
            coef_df['z_score'] = z_scores
            coef_df['p_value'] = p_values
            coef_df['significant'] = p_values < 0.05
        
        # Identify most important features
        top_positive = coef_df[coef_df['coefficient'] > 0].head(10)
        top_negative = coef_df[coef_df['coefficient'] < 0].head(10)
        
        return {
            'intercept': float(intercept),
            'coefficients_df': coef_df.to_dict('records'),
            'top_positive_features': top_positive.to_dict('records'),
            'top_negative_features': top_negative.to_dict('records'),
            'n_significant_features': int(coef_df['significant'].sum()) if 'significant' in coef_df.columns else None,
            'coefficient_statistics': {
                'mean_abs_coefficient': float(np.mean(np.abs(coefficients))),
                'max_abs_coefficient': float(np.max(np.abs(coefficients))),
                'min_abs_coefficient': float(np.min(np.abs(coefficients))),
                'std_coefficient': float(np.std(coefficients))
            }
        }
    
    def _estimate_coefficient_std_errors(self, model: LogisticRegression,
                                       X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Estimate standard errors of coefficients using Fisher Information Matrix."""
        try:
            # Get predicted probabilities
            p = model.predict_proba(X)[:, 1]
            
            # Calculate weights for Fisher Information Matrix
            weights = p * (1 - p)
            
            # Fisher Information Matrix
            X_weighted = X * np.sqrt(weights).reshape(-1, 1)
            fisher_info = np.dot(X_weighted.T, X_weighted)
            
            # Covariance matrix (inverse of Fisher Information)
            cov_matrix = np.linalg.inv(fisher_info)
            
            # Standard errors are square roots of diagonal elements
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            return std_errors
        except Exception as e:
            logger.warning(f"Could not estimate standard errors: {str(e)}")
            return None
    
    def _perform_statistical_tests(self, model: LogisticRegression, X: np.ndarray,
                                 y: np.ndarray, feature_names: list) -> Dict[str, Any]:
        """Perform statistical tests on the model."""
        try:
            # Likelihood ratio test for overall model significance
            null_log_likelihood = self._calculate_null_log_likelihood(y)
            model_log_likelihood = self._calculate_model_log_likelihood(model, X, y)
            
            likelihood_ratio = -2 * (null_log_likelihood - model_log_likelihood)
            df = len(feature_names)
            p_value_lr = 1 - stats.chi2.cdf(likelihood_ratio, df)
            
            # Pseudo R-squared measures
            mcfadden_r2 = 1 - (model_log_likelihood / null_log_likelihood)
            
            # AIC and BIC
            n_params = len(feature_names) + 1  # coefficients + intercept
            n_samples = X.shape[0]
            aic = 2 * n_params - 2 * model_log_likelihood
            bic = np.log(n_samples) * n_params - 2 * model_log_likelihood
            
            return {
                'likelihood_ratio_test': {
                    'statistic': float(likelihood_ratio),
                    'p_value': float(p_value_lr),
                    'degrees_of_freedom': int(df),
                    'significant': bool(p_value_lr < 0.05)
                },
                'goodness_of_fit': {
                    'mcfadden_r2': float(mcfadden_r2),
                    'aic': float(aic),
                    'bic': float(bic),
                    'log_likelihood': float(model_log_likelihood),
                    'null_log_likelihood': float(null_log_likelihood)
                }
            }
        except Exception as e:
            logger.warning(f"Could not perform statistical tests: {str(e)}")
            return {}
    
    def _calculate_null_log_likelihood(self, y: np.ndarray) -> float:
        """Calculate log-likelihood of null model (intercept only)."""
        p1 = np.mean(y)  # Proportion of positive class
        if p1 == 0 or p1 == 1:
            return 0.0
        return np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    
    def _calculate_model_log_likelihood(self, model: LogisticRegression,
                                      X: np.ndarray, y: np.ndarray) -> float:
        """Calculate log-likelihood of fitted model."""
        try:
            proba = model.predict_proba(X)[:, 1]
            # Avoid log(0) by clipping probabilities
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return np.sum(y * np.log(proba) + (1 - y) * np.log(1 - proba))
        except Exception:
            return 0.0
    
    def _save_model(self, model: LogisticRegression, context: Dict[str, Any],
                   hyperparameters: Dict[str, Any]) -> str:
        """Save trained model to disk."""
        try:
            output_dir = context.get('output_dir', 'outputs/models')
            os.makedirs(output_dir, exist_ok=True)
            
            model_filename = hyperparameters.get('model_filename', 'logistic_regression_model.pkl')
            model_path = os.path.join(output_dir, model_filename)
            
            # Save model with metadata
            model_data = {
                'model': model,
                'scaler': self.scaler,
                'feature_names': self.feature_names_,
                'hyperparameters': hyperparameters,
                'coefficients': model.coef_[0],
                'intercept': model.intercept_[0]
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to: {model_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call execute() first.")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call execute() first.")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
