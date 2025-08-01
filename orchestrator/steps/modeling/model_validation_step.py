"""
Model validation step for comprehensive evaluation of trained models.

This step implements cross-validation, performance metrics calculation,
and statistical validation for both classification and regression models.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    validation_curve, learning_curve, permutation_test_score
)
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, matthews_corrcoef, cohen_kappa_score,
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class ModelValidationStep(BaseStep):
    """
    Comprehensive model validation step.
    
    Performs cross-validation, statistical testing, learning curves,
    and detailed performance analysis for trained models.
    """
    
    def __init__(self):
        super().__init__()
        self.validation_results_: Optional[Dict[str, Any]] = None
        self.cross_validation_scores_: Optional[Dict[str, np.ndarray]] = None
        
    def get_step_type(self) -> str:
        return "model_validation"
    
    def get_required_inputs(self) -> list:
        return ['trained_model', 'validation_data']
    
    def get_outputs(self) -> list:
        return ['validation_metrics', 'cross_validation_results', 'statistical_tests', 'validation_plots']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate model validation hyperparameters."""
        # Validate validation type
        validation_type = hyperparameters.get('validation_type', 'classification')
        if validation_type not in ['classification', 'regression']:
            logger.error(f"Invalid validation_type: {validation_type}. Must be 'classification' or 'regression'")
            return False
        
        # Validate cross-validation parameters
        cv_folds = hyperparameters.get('cv_folds', 5)
        if not isinstance(cv_folds, int) or cv_folds < 2 or cv_folds > 20:
            logger.error("cv_folds must be an integer between 2 and 20")
            return False
        
        # Validate scoring metrics
        valid_classification_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'precision_macro',
            'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted'
        ]
        
        valid_regression_metrics = [
            'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2',
            'neg_mean_absolute_percentage_error', 'explained_variance'
        ]
        
        scoring_metrics = hyperparameters.get('scoring_metrics', [])
        if scoring_metrics:
            valid_metrics = valid_classification_metrics if validation_type == 'classification' else valid_regression_metrics
            for metric in scoring_metrics:
                if metric not in valid_metrics:
                    logger.error(f"Invalid scoring metric '{metric}' for {validation_type}")
                    return False
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute comprehensive model validation.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing validation results and outputs
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
            
            # Load trained model
            model_info = context['inputs']['trained_model']
            if isinstance(model_info, dict) and 'model_object' in model_info:
                model = model_info['model_object']
                scaler = model_info.get('scaler_object', None)
                task_type = model_info.get('task_type', hyperparameters.get('validation_type', 'classification'))
            else:
                # Load from file
                model_data = joblib.load(model_info)
                model = model_data['model']
                scaler = model_data.get('scaler', None)
                task_type = model_data.get('task_type', hyperparameters.get('validation_type', 'classification'))
            
            # Load validation data
            validation_data_path = context['inputs']['validation_data']
            validation_data = self._load_validation_data(validation_data_path)
            
            if validation_data is None:
                return {
                    'status': 'failed',
                    'error': 'Failed to load validation data',
                    'outputs': {}
                }
            
            # Prepare features and target
            X, y, feature_names = self._prepare_validation_data(
                validation_data, hyperparameters
            )
            
            # Apply preprocessing if needed
            if scaler is not None:
                X = scaler.transform(X)
            
            # Perform comprehensive validation
            logger.info(f"Starting comprehensive validation for {task_type} model")
            
            # 1. Basic performance metrics
            basic_metrics = self._calculate_basic_metrics(model, X, y, task_type)
            
            # 2. Cross-validation analysis
            cv_results = self._perform_cross_validation(
                model, X, y, task_type, hyperparameters
            )
            
            # 3. Statistical significance tests
            statistical_tests = self._perform_statistical_tests(
                model, X, y, task_type, hyperparameters
            )
            
            # 4. Learning curves
            learning_curves = self._generate_learning_curves(
                model, X, y, task_type, hyperparameters
            )
            
            # 5. Validation curves for hyperparameters
            validation_curves = self._generate_validation_curves(
                model, X, y, task_type, hyperparameters
            )
            
            # 6. Feature importance validation
            feature_validation = self._validate_feature_importance(
                model, X, y, feature_names, task_type, hyperparameters
            )
            
            # 7. Model calibration (for classification)
            calibration_results = None
            if task_type == 'classification':
                calibration_results = self._analyze_model_calibration(model, X, y)
            
            # 8. Generate validation plots
            validation_plots = self._generate_validation_plots(
                model, X, y, task_type, basic_metrics, cv_results, context
            )
            
            # Store results
            self.validation_results_ = {
                'basic_metrics': basic_metrics,
                'cross_validation': cv_results,
                'statistical_tests': statistical_tests,
                'learning_curves': learning_curves,
                'validation_curves': validation_curves,
                'feature_validation': feature_validation,
                'calibration_results': calibration_results
            }
            
            # Prepare outputs
            outputs = {
                'validation_metrics': basic_metrics,
                'cross_validation_results': cv_results,
                'statistical_tests': statistical_tests,
                'validation_plots': validation_plots,
                'learning_curves': learning_curves,
                'validation_curves': validation_curves,
                'feature_validation': feature_validation
            }
            
            if calibration_results:
                outputs['calibration_results'] = calibration_results
            
            # Generate validation summary
            validation_summary = self._generate_validation_summary(
                basic_metrics, cv_results, statistical_tests, task_type
            )
            
            logger.info("Model validation completed successfully")
            logger.info(f"Validation summary: {validation_summary['overall_assessment']}")
            
            return {
                'status': 'success',
                'message': 'Model validation completed successfully',
                'outputs': outputs,
                'validation_summary': validation_summary,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'validation_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'task_type': task_type
                }
            }
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_validation_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load validation data from file."""
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
            logger.error(f"Failed to load validation data: {str(e)}")
            return None
    
    def _prepare_validation_data(self, data: pd.DataFrame, 
                               hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prepare validation features and target."""
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
        
        logger.info(f"Prepared validation data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_columns
    
    def _calculate_basic_metrics(self, model, X: np.ndarray, y: np.ndarray, 
                               task_type: str) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        if task_type == 'classification':
            return self._calculate_classification_metrics(model, X, y)
        else:
            return self._calculate_regression_metrics(model, X, y)
    
    def _calculate_classification_metrics(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        # Additional metrics for binary classification
        if len(np.unique(y)) == 2:
            precision_binary = precision_score(y, y_pred, zero_division=0)
            recall_binary = recall_score(y, y_pred, zero_division=0)
            f1_binary = f1_score(y, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y, y_pred)
            kappa = cohen_kappa_score(y, y_pred)
            
            metrics.update({
                'precision_binary': float(precision_binary),
                'recall_binary': float(recall_binary),
                'f1_score_binary': float(f1_binary),
                'matthews_corrcoef': float(mcc),
                'cohen_kappa': float(kappa)
            })
            
            if y_pred_proba is not None:
                auc = roc_auc_score(y, y_pred_proba)
                fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
                
                metrics.update({
                    'roc_auc': float(auc),
                    'roc_curve': {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist()
                    }
                })
        
        return metrics
    
    def _calculate_regression_metrics(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive regression metrics."""
        y_pred = model.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        explained_var = explained_variance_score(y, y_pred)
        
        # Additional metrics
        try:
            mape = mean_absolute_percentage_error(y, y_pred)
        except:
            mape = np.mean(np.abs((y - y_pred) / np.where(y != 0, y, 1))) * 100
        
        # Residual analysis
        residuals = y - y_pred
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'explained_variance': float(explained_var),
            'mape': float(mape),
            'residual_statistics': {
                'mean': float(residual_mean),
                'std': float(residual_std),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals))
            },
            'prediction_statistics': {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred))
            }
        }
    
    def _perform_cross_validation(self, model, X: np.ndarray, y: np.ndarray,
                                task_type: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive cross-validation analysis."""
        cv_folds = hyperparameters.get('cv_folds', 5)
        random_state = hyperparameters.get('random_state', 42)
        
        # Choose cross-validation strategy
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            default_scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            if len(np.unique(y)) == 2:
                default_scoring.append('roc_auc')
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            default_scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        # Get scoring metrics
        scoring_metrics = hyperparameters.get('scoring_metrics', default_scoring)
        
        # Perform cross-validation
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring_metrics,
            return_train_score=True, return_estimator=False
        )
        
        # Process results
        processed_results = {}
        for metric in scoring_metrics:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            processed_results[metric] = {
                'test_scores': test_scores.tolist(),
                'train_scores': train_scores.tolist(),
                'test_mean': float(np.mean(test_scores)),
                'test_std': float(np.std(test_scores)),
                'train_mean': float(np.mean(train_scores)),
                'train_std': float(np.std(train_scores)),
                'overfitting_score': float(np.mean(train_scores) - np.mean(test_scores))
            }
        
        # Store cross-validation scores
        self.cross_validation_scores_ = {
            metric: cv_results[f'test_{metric}'] for metric in scoring_metrics
        }
        
        # Calculate overall cross-validation statistics
        primary_metric = scoring_metrics[0]
        cv_summary = {
            'cv_folds': cv_folds,
            'primary_metric': primary_metric,
            'primary_score_mean': processed_results[primary_metric]['test_mean'],
            'primary_score_std': processed_results[primary_metric]['test_std'],
            'confidence_interval_95': self._calculate_confidence_interval(
                processed_results[primary_metric]['test_scores']
            )
        }
        
        return {
            'cv_results': processed_results,
            'cv_summary': cv_summary,
            'scoring_metrics': scoring_metrics
        }
    
    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for cross-validation scores."""
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_error = stats.sem(scores_array)
        
        # Use t-distribution for small samples
        df = len(scores_array) - 1
        t_value = stats.t.ppf((1 + confidence) / 2, df)
        
        margin_error = t_value * std_error
        return (float(mean_score - margin_error), float(mean_score + margin_error))
    
    def _perform_statistical_tests(self, model, X: np.ndarray, y: np.ndarray,
                                 task_type: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        statistical_tests = {}
        
        try:
            # Permutation test for model significance
            logger.info("Performing permutation test for model significance")
            
            cv_folds = hyperparameters.get('cv_folds', 5)
            n_permutations = hyperparameters.get('n_permutations', 100)
            random_state = hyperparameters.get('random_state', 42)
            
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'r2'
            
            score, permutation_scores, p_value = permutation_test_score(
                model, X, y, cv=cv, scoring=scoring, 
                n_permutations=n_permutations, random_state=random_state
            )
            
            statistical_tests['permutation_test'] = {
                'score': float(score),
                'permutation_scores': permutation_scores.tolist(),
                'p_value': float(p_value),
                'n_permutations': n_permutations,
                'significant': bool(p_value < 0.05)
            }
            
        except Exception as e:
            logger.warning(f"Permutation test failed: {str(e)}")
            statistical_tests['permutation_test'] = {'error': str(e)}
        
        # Additional statistical tests based on cross-validation results
        if self.cross_validation_scores_ is not None:
            primary_metric = list(self.cross_validation_scores_.keys())[0]
            cv_scores = self.cross_validation_scores_[primary_metric]
            
            # One-sample t-test against chance performance
            if task_type == 'classification':
                # For balanced binary classification, chance = 0.5
                chance_performance = 0.5 if len(np.unique(y)) == 2 else 1.0 / len(np.unique(y))
            else:
                # For regression, test against R² = 0 (no predictive power)
                chance_performance = 0.0
            
            t_stat, t_p_value = stats.ttest_1samp(cv_scores, chance_performance)
            
            statistical_tests['t_test_vs_chance'] = {
                'chance_performance': float(chance_performance),
                't_statistic': float(t_stat),
                'p_value': float(t_p_value),
                'significant': bool(t_p_value < 0.05)
            }
        
        return statistical_tests
    
    def _generate_learning_curves(self, model, X: np.ndarray, y: np.ndarray,
                                task_type: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate learning curves to analyze model performance vs training set size."""
        try:
            cv_folds = hyperparameters.get('cv_folds', 5)
            random_state = hyperparameters.get('random_state', 42)
            
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'r2'
            
            # Define training sizes
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            logger.info("Generating learning curves")
            
            train_sizes_abs, train_scores, validation_scores = learning_curve(
                model, X, y, cv=cv, train_sizes=train_sizes,
                scoring=scoring, random_state=random_state
            )
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores': {
                    'mean': np.mean(train_scores, axis=1).tolist(),
                    'std': np.std(train_scores, axis=1).tolist(),
                    'all_scores': train_scores.tolist()
                },
                'validation_scores': {
                    'mean': np.mean(validation_scores, axis=1).tolist(),
                    'std': np.std(validation_scores, axis=1).tolist(),
                    'all_scores': validation_scores.tolist()
                },
                'scoring_metric': scoring
            }
            
        except Exception as e:
            logger.warning(f"Learning curve generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_validation_curves(self, model, X: np.ndarray, y: np.ndarray,
                                  task_type: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation curves for hyperparameter analysis."""
        validation_curves = {}
        
        try:
            cv_folds = hyperparameters.get('cv_folds', 5)
            random_state = hyperparameters.get('random_state', 42)
            
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'r2'
            
            # Define parameters to validate based on model type
            param_ranges = self._get_validation_param_ranges(model)
            
            for param_name, param_range in param_ranges.items():
                logger.info(f"Generating validation curve for {param_name}")
                
                train_scores, validation_scores = validation_curve(
                    model, X, y, param_name=param_name, param_range=param_range,
                    cv=cv, scoring=scoring
                )
                
                validation_curves[param_name] = {
                    'param_range': param_range,
                    'train_scores': {
                        'mean': np.mean(train_scores, axis=1).tolist(),
                        'std': np.std(train_scores, axis=1).tolist()
                    },
                    'validation_scores': {
                        'mean': np.mean(validation_scores, axis=1).tolist(),
                        'std': np.std(validation_scores, axis=1).tolist()
                    },
                    'optimal_param': param_range[np.argmax(np.mean(validation_scores, axis=1))]
                }
                
        except Exception as e:
            logger.warning(f"Validation curve generation failed: {str(e)}")
            validation_curves['error'] = str(e)
        
        return validation_curves
    
    def _get_validation_param_ranges(self, model) -> Dict[str, List]:
        """Get parameter ranges for validation curves based on model type."""
        model_name = model.__class__.__name__
        
        if 'RandomForest' in model_name:
            return {
                'n_estimators': [10, 50, 100, 200, 500],
                'max_depth': [3, 5, 10, 15, None]
            }
        elif 'LogisticRegression' in model_name:
            return {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        elif 'SVC' in model_name or 'SVR' in model_name:
            return {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
            }
        else:
            return {}
    
    def _validate_feature_importance(self, model, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str], task_type: str,
                                   hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature importance through permutation testing."""
        try:
            from sklearn.inspection import permutation_importance
            
            logger.info("Calculating permutation-based feature importance")
            
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            
            perm_importance = permutation_importance(
                model, X, y, scoring=scoring,
                n_repeats=hyperparameters.get('n_permutation_repeats', 10),
                random_state=hyperparameters.get('random_state', 42)
            )
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            # Calculate statistical significance of feature importance
            importance_df['importance_pvalue'] = [
                2 * (1 - stats.norm.cdf(abs(mean / std))) if std > 0 else 1.0
                for mean, std in zip(perm_importance.importances_mean, perm_importance.importances_std)
            ]
            
            importance_df['significant'] = importance_df['importance_pvalue'] < 0.05
            
            return {
                'permutation_importance': {
                    'importances_mean': perm_importance.importances_mean.tolist(),
                    'importances_std': perm_importance.importances_std.tolist(),
                    'importances_all': perm_importance.importances.tolist()
                },
                'feature_importance_df': importance_df.to_dict('records'),
                'significant_features': importance_df[importance_df['significant']].to_dict('records'),
                'n_significant_features': int(importance_df['significant'].sum())
            }
            
        except Exception as e:
            logger.warning(f"Feature importance validation failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_model_calibration(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze model calibration for classification models."""
        try:
            if not hasattr(model, 'predict_proba'):
                return {'error': 'Model does not support probability prediction'}
            
            y_proba = model.predict_proba(X)[:, 1]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y, y_proba, n_bins=10
            )
            
            # Calculate Brier score
            brier_score = np.mean((y_proba - y) ** 2)
            
            return {
                'calibration_curve': {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                },
                'brier_score': float(brier_score),
                'perfectly_calibrated': np.allclose(fraction_of_positives, mean_predicted_value, atol=0.1)
            }
            
        except Exception as e:
            logger.warning(f"Model calibration analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_validation_plots(self, model, X: np.ndarray, y: np.ndarray,
                                 task_type: str, basic_metrics: Dict[str, Any],
                                 cv_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation plots and save them."""
        plots = {}
        
        try:
            output_dir = context.get('output_dir', 'outputs/validation')
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Confusion Matrix (for classification)
            if task_type == 'classification' and 'confusion_matrix' in basic_metrics:
                confusion_plot_path = self._plot_confusion_matrix(
                    basic_metrics['confusion_matrix'], output_dir
                )
                plots['confusion_matrix'] = confusion_plot_path
            
            # 2. Cross-validation scores plot
            cv_plot_path = self._plot_cv_scores(cv_results, output_dir)
            plots['cross_validation_scores'] = cv_plot_path
            
            # 3. ROC Curve (for binary classification)
            if task_type == 'classification' and 'roc_curve' in basic_metrics:
                roc_plot_path = self._plot_roc_curve(basic_metrics['roc_curve'], output_dir)
                plots['roc_curve'] = roc_plot_path
            
            # 4. Residual plots (for regression)
            if task_type == 'regression':
                y_pred = model.predict(X)
                residual_plot_path = self._plot_residuals(y, y_pred, output_dir)
                plots['residual_plot'] = residual_plot_path
                
                prediction_plot_path = self._plot_predictions(y, y_pred, output_dir)
                plots['prediction_plot'] = prediction_plot_path
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {str(e)}")
            plots['error'] = str(e)
        
        return plots
    
    def _plot_confusion_matrix(self, confusion_matrix: List[List[int]], output_dir: str) -> str:
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_cv_scores(self, cv_results: Dict[str, Any], output_dir: str) -> str:
        """Plot cross-validation scores."""
        plt.figure(figsize=(12, 8))
        
        cv_data = cv_results['cv_results']
        metrics = list(cv_data.keys())
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            scores = cv_data[metric]['test_scores']
            plt.bar(range(len(scores)), scores)
            plt.title(f'{metric} - CV Scores')
            plt.xlabel('Fold')
            plt.ylabel('Score')
            plt.ylim(0, max(scores) * 1.1)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'cv_scores.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_roc_curve(self, roc_data: Dict[str, List], output_dir: str) -> str:
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(roc_data['fpr'], roc_data['tpr'], label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> str:
        """Plot residual analysis for regression."""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 4))
        
        # Residuals vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.axhline(y=0, color='r', linestyle='--')
        
        # Histogram of residuals
        plt.subplot(1, 3, 2)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        
        # Q-Q plot
        plt.subplot(1, 3, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'residual_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> str:
        """Plot predicted vs actual values for regression."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.grid(True)
        
        plot_path = os.path.join(output_dir, 'predicted_vs_actual.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _generate_validation_summary(self, basic_metrics: Dict[str, Any],
                                   cv_results: Dict[str, Any], statistical_tests: Dict[str, Any],
                                   task_type: str) -> Dict[str, Any]:
        """Generate overall validation summary and assessment."""
        summary = {
            'task_type': task_type,
            'overall_assessment': '',
            'key_findings': [],
            'recommendations': []
        }
        
        # Primary metric assessment
        if task_type == 'classification':
            primary_score = basic_metrics.get('accuracy', 0)
            cv_score = cv_results['cv_summary']['primary_score_mean']
            score_name = 'Accuracy'
        else:
            primary_score = basic_metrics.get('r2_score', 0)
            cv_score = cv_results['cv_summary']['primary_score_mean']
            score_name = 'R² Score'
        
        # Performance assessment
        if primary_score > 0.9:
            performance_level = 'Excellent'
        elif primary_score > 0.8:
            performance_level = 'Good'
        elif primary_score > 0.7:
            performance_level = 'Fair'
        else:
            performance_level = 'Poor'
        
        summary['key_findings'].append(f"{score_name}: {primary_score:.3f} ({performance_level})")
        summary['key_findings'].append(f"Cross-validation {score_name}: {cv_score:.3f} ± {cv_results['cv_summary']['primary_score_std']:.3f}")
        
        # Overfitting assessment
        overfitting_score = abs(cv_results['cv_results'][cv_results['cv_summary']['primary_metric']]['overfitting_score'])
        if overfitting_score > 0.1:
            summary['key_findings'].append(f"High overfitting detected (difference: {overfitting_score:.3f})")
            summary['recommendations'].append("Consider regularization or reducing model complexity")
        elif overfitting_score > 0.05:
            summary['key_findings'].append(f"Moderate overfitting detected (difference: {overfitting_score:.3f})")
            summary['recommendations'].append("Monitor model complexity")
        else:
            summary['key_findings'].append("Good generalization (low overfitting)")
        
        # Statistical significance
        if 'permutation_test' in statistical_tests and 'p_value' in statistical_tests['permutation_test']:
            p_value = statistical_tests['permutation_test']['p_value']
            if p_value < 0.001:
                summary['key_findings'].append("Model is highly statistically significant (p < 0.001)")
            elif p_value < 0.05:
                summary['key_findings'].append(f"Model is statistically significant (p = {p_value:.3f})")
            else:
                summary['key_findings'].append(f"Model significance is questionable (p = {p_value:.3f})")
                summary['recommendations'].append("Consider collecting more data or improving features")
        
        # Overall assessment
        if performance_level in ['Excellent', 'Good'] and overfitting_score < 0.1:
            summary['overall_assessment'] = 'Model shows strong performance with good generalization'
        elif performance_level in ['Excellent', 'Good']:
            summary['overall_assessment'] = 'Model shows good performance but may be overfitting'
        elif performance_level == 'Fair':
            summary['overall_assessment'] = 'Model shows moderate performance with room for improvement'
        else:
            summary['overall_assessment'] = 'Model performance is poor and requires significant improvement'
        
        return summary
    
    def get_validation_results(self) -> Optional[Dict[str, Any]]:
        """Get complete validation results."""
        return self.validation_results_
    
    def get_cross_validation_scores(self) -> Optional[Dict[str, np.ndarray]]:
        """Get cross-validation scores for all metrics."""
        return self.cross_validation_scores_
