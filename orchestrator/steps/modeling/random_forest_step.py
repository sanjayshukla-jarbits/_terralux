"""
Random Forest modeling step for landslide susceptibility and mineral targeting.

This step implements Random Forest classification and regression using scikit-learn,
with comprehensive hyperparameter configuration and model persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, Optional, Union, Tuple
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class RandomForestStep(BaseStep):
    """
    Random Forest training and prediction step.
    
    Supports both classification (landslide susceptibility) and regression
    (mineral prospectivity scores) tasks with comprehensive configuration options.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[Union[RandomForestClassifier, RandomForestRegressor]] = None
        self.feature_importance_: Optional[np.ndarray] = None
        self.feature_names_: Optional[list] = None
        
    def get_step_type(self) -> str:
        return "random_forest_training"
    
    def get_required_inputs(self) -> list:
        return ['training_data']
    
    def get_outputs(self) -> list:
        return ['trained_model', 'model_metrics', 'feature_importance']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate Random Forest hyperparameters."""
        required_params = ['task_type']  # 'classification' or 'regression'
        
        for param in required_params:
            if param not in hyperparameters:
                logger.error(f"Missing required hyperparameter: {param}")
                return False
        
        task_type = hyperparameters['task_type']
        if task_type not in ['classification', 'regression']:
            logger.error(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
            return False
            
        # Validate numeric parameters
        numeric_params = {
            'n_estimators': (int, 1, 1000),
            'max_depth': (int, 1, 100),
            'min_samples_split': (int, 2, 100),
            'min_samples_leaf': (int, 1, 50),
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
        Execute Random Forest training.
        
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
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if hyperparameters['task_type'] == 'classification' else None
            )
            
            # Create and train model
            model = self._create_model(hyperparameters)
            
            logger.info(f"Training Random Forest with {X_train.shape[0]} samples, {X_train.shape[1]} features")
            model.fit(X_train, y_train)
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(
                y_test, y_pred, hyperparameters['task_type']
            )
            
            # Store model and feature information
            self.model = model
            self.feature_importance_ = model.feature_importances_
            self.feature_names_ = feature_names
            
            # Save model if requested
            model_path = None
            if hyperparameters.get('save_model', True):
                model_path = self._save_model(model, context, hyperparameters)
            
            # Create feature importance analysis
            feature_importance_df = self._create_feature_importance_analysis(
                model.feature_importances_, feature_names
            )
            
            # Prepare outputs
            outputs = {
                'trained_model': {
                    'model_object': model,
                    'model_path': model_path,
                    'task_type': hyperparameters['task_type'],
                    'n_features': len(feature_names),
                    'n_estimators': model.n_estimators
                },
                'model_metrics': metrics,
                'feature_importance': {
                    'importance_scores': model.feature_importances_,
                    'feature_names': feature_names,
                    'importance_df': feature_importance_df,
                    'top_features': feature_importance_df.head(20).to_dict('records')
                }
            }
            
            logger.info(f"Random Forest training completed successfully")
            logger.info(f"Model performance: {metrics}")
            
            return {
                'status': 'success',
                'message': f'Random Forest {hyperparameters["task_type"]} model trained successfully',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'training_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'n_features': X_train.shape[1]
                }
            }
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
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
        y = data_clean[target_column].values
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Target distribution: {np.bincount(y.astype(int)) if hyperparameters['task_type'] == 'classification' else f'Range: {y.min():.3f} - {y.max():.3f}'}")
        
        return X, y, feature_columns
    
    def _create_model(self, hyperparameters: Dict[str, Any]) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """Create Random Forest model based on task type."""
        common_params = {
            'n_estimators': hyperparameters.get('n_estimators', 100),
            'max_depth': hyperparameters.get('max_depth', None),
            'min_samples_split': hyperparameters.get('min_samples_split', 2),
            'min_samples_leaf': hyperparameters.get('min_samples_leaf', 1),
            'max_features': hyperparameters.get('max_features', 'sqrt'),
            'random_state': hyperparameters.get('random_state', 42),
            'n_jobs': hyperparameters.get('n_jobs', -1),
            'verbose': hyperparameters.get('verbose', 0)
        }
        
        if hyperparameters['task_type'] == 'classification':
            additional_params = {
                'class_weight': hyperparameters.get('class_weight', 'balanced'),
                'criterion': hyperparameters.get('criterion', 'gini')
            }
            return RandomForestClassifier(**common_params, **additional_params)
        else:
            additional_params = {
                'criterion': hyperparameters.get('criterion', 'squared_error')
            }
            return RandomForestRegressor(**common_params, **additional_params)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Calculate performance metrics based on task type."""
        if task_type == 'classification':
            return self._calculate_classification_metrics(y_true, y_pred)
        else:
            return self._calculate_regression_metrics(y_true, y_pred)
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'precision': float(class_report['weighted avg']['precision']),
            'recall': float(class_report['weighted avg']['recall']),
            'f1_score': float(class_report['weighted avg']['f1-score'])
        }
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mean_prediction': float(np.mean(y_pred)),
            'std_prediction': float(np.std(y_pred))
        }
    
    def _save_model(self, model: Union[RandomForestClassifier, RandomForestRegressor], 
                   context: Dict[str, Any], hyperparameters: Dict[str, Any]) -> str:
        """Save trained model to disk."""
        try:
            output_dir = context.get('output_dir', 'outputs/models')
            os.makedirs(output_dir, exist_ok=True)
            
            model_filename = hyperparameters.get('model_filename', 
                f"random_forest_{hyperparameters['task_type']}_model.pkl")
            model_path = os.path.join(output_dir, model_filename)
            
            # Save model with metadata
            model_data = {
                'model': model,
                'feature_names': self.feature_names_,
                'task_type': hyperparameters['task_type'],
                'hyperparameters': hyperparameters,
                'feature_importance': model.feature_importances_
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to: {model_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return None
    
    def _create_feature_importance_analysis(self, importance_scores: np.ndarray, 
                                          feature_names: list) -> pd.DataFrame:
        """Create feature importance analysis DataFrame."""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
        importance_df['cumulative_importance'] = importance_df['importance_normalized'].cumsum()
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call execute() first.")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance analysis."""
        if self.feature_importance_ is None or self.feature_names_ is None:
            return None
        return self._create_feature_importance_analysis(
            self.feature_importance_, self.feature_names_
        )
