"""
Statistical plots and performance visualization step.

This step creates comprehensive statistical visualizations including performance
metrics, model comparisons, feature importance, and validation plots for both
landslide susceptibility and mineral targeting applications.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, mean_squared_error, r2_score
)
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class StatisticalPlotsStep(BaseStep):
    """
    Statistical plots and performance visualization step.
    
    Creates comprehensive statistical visualizations for model performance,
    feature analysis, and validation results with domain-specific interpretations.
    """
    
    def __init__(self):
        super().__init__()
        self.plot_objects_: Dict[str, Any] = {}
        self.plot_metadata_: Dict[str, Any] = {}
        
    def get_step_type(self) -> str:
        return "statistical_plots"
    
    def get_required_inputs(self) -> list:
        return ['model_results']
    
    def get_outputs(self) -> list:
        return ['statistical_plots', 'plot_metadata']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate statistical plots hyperparameters."""
        # Validate plot types
        plot_types = hyperparameters.get('plot_types', ['performance', 'feature_importance'])
        valid_types = [
            'performance', 'feature_importance', 'confusion_matrix', 'roc_curve',
            'precision_recall', 'residuals', 'distribution', 'correlation',
            'validation_curve', 'learning_curve', 'model_comparison'
        ]
        
        for plot_type in plot_types:
            if plot_type not in valid_types:
                logger.error(f"Invalid plot_type: {plot_type}. Must be one of {valid_types}")
                return False
        
        # Validate plot format
        plot_format = hyperparameters.get('plot_format', 'png')
        valid_formats = ['png', 'pdf', 'svg', 'html', 'both']
        if plot_format not in valid_formats:
            logger.error(f"Invalid plot_format: {plot_format}. Must be one of {valid_formats}")
            return False
        
        # Validate figure size
        figure_size = hyperparameters.get('figure_size', [12, 8])
        if not isinstance(figure_size, list) or len(figure_size) != 2:
            logger.error("figure_size must be a list of [width, height]")
            return False
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute statistical plots generation.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing plot outputs and metadata
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
            
            # Load model results
            model_results = self._load_model_results(
                context['inputs']['model_results'], hyperparameters
            )
            
            if not model_results:
                return {
                    'status': 'failed',
                    'error': 'Failed to load model results',
                    'outputs': {}
                }
            
            # Set up plotting environment
            self._setup_plotting_style(hyperparameters)
            
            # Generate requested plots
            plot_types = hyperparameters.get('plot_types', ['performance', 'feature_importance'])
            generated_plots = {}
            
            for plot_type in plot_types:
                try:
                    plots = self._generate_plot_type(plot_type, model_results, hyperparameters, context)
                    if plots:
                        generated_plots[plot_type] = plots
                except Exception as e:
                    logger.warning(f"Failed to generate {plot_type} plots: {str(e)}")
            
            # Create summary dashboard if requested
            if hyperparameters.get('create_dashboard', True):
                dashboard = self._create_summary_dashboard(
                    model_results, generated_plots, hyperparameters, context
                )
                if dashboard:
                    generated_plots['dashboard'] = dashboard
            
            # Generate plot metadata
            plot_metadata = self._generate_plot_metadata(
                generated_plots, model_results, hyperparameters
            )
            
            # Store results
            self.plot_objects_ = generated_plots
            self.plot_metadata_ = plot_metadata
            
            # Prepare outputs
            outputs = {
                'statistical_plots': generated_plots,
                'plot_metadata': plot_metadata
            }
            
            logger.info("Statistical plots generation completed successfully")
            logger.info(f"Generated {len(generated_plots)} plot types")
            
            return {
                'status': 'success',
                'message': 'Statistical plots created successfully',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'n_plot_types': len(generated_plots),
                    'plot_format': hyperparameters.get('plot_format', 'png'),
                    'application_type': hyperparameters.get('application_type', 'generic')
                }
            }
            
        except Exception as e:
            logger.error(f"Statistical plots generation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_model_results(self, model_input: Union[str, Dict[str, Any]], 
                          hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Load model results from various sources."""
        try:
            if isinstance(model_input, dict):
                return model_input
            elif isinstance(model_input, str):
                # Load from file
                if model_input.endswith('.json'):
                    import json
                    with open(model_input, 'r') as f:
                        return json.load(f)
                elif model_input.endswith('.pkl'):
                    import joblib
                    return joblib.load(model_input)
            
            logger.error(f"Unsupported model results format: {type(model_input)}")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to load model results: {str(e)}")
            return {}
    
    def _setup_plotting_style(self, hyperparameters: Dict[str, Any]):
        """Setup plotting style and theme."""
        # Set matplotlib style
        plt.style.use(hyperparameters.get('plot_style', 'seaborn-v0_8'))
        
        # Set color palette
        application_type = hyperparameters.get('application_type', 'generic')
        
        if application_type == 'landslide_susceptibility':
            colors = ['#2E8B57', '#FFD700', '#FF8C00', '#FF4500', '#DC143C']  # Green to Red
        elif application_type == 'mineral_targeting':
            colors = ['#2F4F4F', '#4682B4', '#DAA520', '#DC143C']  # Dark to Bright
        else:
            colors = sns.color_palette("viridis", 5)
        
        sns.set_palette(colors)
        
        # Set figure parameters
        plt.rcParams['figure.figsize'] = hyperparameters.get('figure_size', [12, 8])
        plt.rcParams['font.size'] = hyperparameters.get('font_size', 12)
        plt.rcParams['axes.titlesize'] = hyperparameters.get('title_size', 14)
        plt.rcParams['axes.labelsize'] = hyperparameters.get('label_size', 12)
    
    def _generate_plot_type(self, plot_type: str, model_results: Dict[str, Any],
                          hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Generate specific type of plots."""
        if plot_type == 'performance':
            return self._create_performance_plots(model_results, hyperparameters, context)
        elif plot_type == 'feature_importance':
            return self._create_feature_importance_plots(model_results, hyperparameters, context)
        elif plot_type == 'confusion_matrix':
            return self._create_confusion_matrix_plots(model_results, hyperparameters, context)
        elif plot_type == 'roc_curve':
            return self._create_roc_curve_plots(model_results, hyperparameters, context)
        elif plot_type == 'precision_recall':
            return self._create_precision_recall_plots(model_results, hyperparameters, context)
        elif plot_type == 'residuals':
            return self._create_residual_plots(model_results, hyperparameters, context)
        elif plot_type == 'distribution':
            return self._create_distribution_plots(model_results, hyperparameters, context)
        elif plot_type == 'correlation':
            return self._create_correlation_plots(model_results, hyperparameters, context)
        elif plot_type == 'validation_curve':
            return self._create_validation_curve_plots(model_results, hyperparameters, context)
        elif plot_type == 'learning_curve':
            return self._create_learning_curve_plots(model_results, hyperparameters, context)
        elif plot_type == 'model_comparison':
            return self._create_model_comparison_plots(model_results, hyperparameters, context)
        else:
            logger.warning(f"Unknown plot type: {plot_type}")
            return {}
    
    def _create_performance_plots(self, model_results: Dict[str, Any],
                                hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create model performance overview plots."""
        plots = {}
        output_dir = self._get_output_dir(context, 'performance')
        
        try:
            # Extract performance metrics
            metrics = model_results.get('model_metrics', {})
            if not metrics:
                logger.warning("No model metrics found for performance plots")
                return plots
            
            # Create multi-panel performance overview
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Overview', fontsize=16, fontweight='bold')
            
            # Panel 1: Accuracy metrics bar plot
            ax1 = axes[0, 0]
            if 'accuracy' in metrics:
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                metric_values = [
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1_score', 0)
                ]
                
                bars = ax1.bar(metric_names, metric_values, alpha=0.7)
                ax1.set_ylim(0, 1)
                ax1.set_ylabel('Score')
                ax1.set_title('Classification Metrics')
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            
            # Panel 2: Regression metrics (if available)
            ax2 = axes[0, 1]
            if 'r2_score' in metrics:
                reg_names = ['R²', 'RMSE', 'MAE']
                reg_values = [
                    metrics.get('r2_score', 0),
                    metrics.get('rmse', 0),
                    metrics.get('mae', 0)
                ]
                
                # Normalize RMSE and MAE for display
                if reg_values[1] > 0:
                    reg_values[1] = 1 - min(reg_values[1], 1)  # Convert to "goodness" metric
                if reg_values[2] > 0:
                    reg_values[2] = 1 - min(reg_values[2], 1)  # Convert to "goodness" metric
                
                bars = ax2.bar(reg_names, reg_values, alpha=0.7, color='orange')
                ax2.set_ylim(0, 1)
                ax2.set_ylabel('Score')
                ax2.set_title('Regression Metrics')
            else:
                ax2.text(0.5, 0.5, 'No regression metrics available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Regression Metrics')
            
            # Panel 3: Cross-validation results (if available)
            ax3 = axes[1, 0]
            cv_results = model_results.get('cross_validation_results', {})
            if cv_results and 'cv_results' in cv_results:
                cv_data = cv_results['cv_results']
                metric_name = list(cv_data.keys())[0]  # Use first metric
                scores = cv_data[metric_name]['test_scores']
                
                ax3.boxplot([scores], labels=[metric_name])
                ax3.set_ylabel('Score')
                ax3.set_title(f'Cross-Validation Results ({len(scores)} folds)')
                
                # Add individual points
                ax3.scatter([1] * len(scores), scores, alpha=0.5, color='red')
            else:
                ax3.text(0.5, 0.5, 'No CV results available',
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Cross-Validation Results')
            
            # Panel 4: Feature importance (top 10)
            ax4 = axes[1, 1]
            feature_importance = model_results.get('feature_importance', {})
            if feature_importance and 'importance_scores' in feature_importance:
                importance = np.array(feature_importance['importance_scores'])
                feature_names = feature_importance.get('feature_names', [f'Feature_{i}' for i in range(len(importance))])
                
                # Get top 10 features
                top_indices = np.argsort(importance)[-10:][::-1]
                top_importance = importance[top_indices]
                top_names = [feature_names[i] for i in top_indices]
                
                y_pos = np.arange(len(top_names))
                ax4.barh(y_pos, top_importance, alpha=0.7)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(top_names)
                ax4.set_xlabel('Importance')
                ax4.set_title('Top 10 Feature Importance')
            else:
                ax4.text(0.5, 0.5, 'No feature importance available',
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Feature Importance')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self._save_plot(fig, output_dir, 'performance_overview', hyperparameters)
            plots['performance_overview'] = plot_path
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create performance plots: {str(e)}")
        
        return plots
    
    def _create_feature_importance_plots(self, model_results: Dict[str, Any],
                                       hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create feature importance visualization plots."""
        plots = {}
        output_dir = self._get_output_dir(context, 'feature_importance')
        
        try:
            feature_importance = model_results.get('feature_importance', {})
            if not feature_importance or 'importance_scores' not in feature_importance:
                logger.warning("No feature importance data found")
                return plots
            
            importance = np.array(feature_importance['importance_scores'])
            feature_names = feature_importance.get('feature_names', [f'Feature_{i}' for i in range(len(importance))])
            
            # Create multiple feature importance plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Top 20 features horizontal bar plot
            ax1 = axes[0, 0]
            top_indices = np.argsort(importance)[-20:]
            top_importance = importance[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            y_pos = np.arange(len(top_names))
            bars = ax1.barh(y_pos, top_importance, alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_names, fontsize=10)
            ax1.set_xlabel('Importance Score')
            ax1.set_title('Top 20 Features')
            
            # Color bars by importance level
            colors = plt.cm.viridis(top_importance / np.max(top_importance))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Plot 2: Feature importance distribution
            ax2 = axes[0, 1]
            ax2.hist(importance, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(importance), color='red', linestyle='--', label=f'Mean: {np.mean(importance):.3f}')
            ax2.axvline(np.median(importance), color='orange', linestyle='--', label=f'Median: {np.median(importance):.3f}')
            ax2.set_xlabel('Importance Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Feature Importance Distribution')
            ax2.legend()
            
            # Plot 3: Cumulative importance
            ax3 = axes[1, 0]
            sorted_importance = np.sort(importance)[::-1]
            cumulative_importance = np.cumsum(sorted_importance) / np.sum(importance)
            
            ax3.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, marker='o', markersize=2)
            ax3.axhline(0.8, color='red', linestyle='--', label='80% threshold')
            ax3.axhline(0.9, color='orange', linestyle='--', label='90% threshold')
            ax3.set_xlabel('Number of Features')
            ax3.set_ylabel('Cumulative Importance')
            ax3.set_title('Cumulative Feature Importance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Feature groups (if feature names suggest groups)
            ax4 = axes[1, 1]
            feature_groups = self._group_features_by_type(feature_names, importance)
            
            if len(feature_groups) > 1:
                group_names = list(feature_groups.keys())
                group_importance = [np.mean(feature_groups[group]) for group in group_names]
                
                wedges, texts, autotexts = ax4.pie(group_importance, labels=group_names, autopct='%1.1f%%',
                                                  startangle=90)
                ax4.set_title('Feature Importance by Group')
            else:
                ax4.text(0.5, 0.5, 'No feature groups identified',
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Feature Groups')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self._save_plot(fig, output_dir, 'feature_importance_analysis', hyperparameters)
            plots['feature_importance_analysis'] = plot_path
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create feature importance plots: {str(e)}")
        
        return plots
    
    def _create_confusion_matrix_plots(self, model_results: Dict[str, Any],
                                     hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create confusion matrix visualization plots."""
        plots = {}
        output_dir = self._get_output_dir(context, 'confusion_matrix')
        
        try:
            metrics = model_results.get('model_metrics', {})
            confusion_mat = metrics.get('confusion_matrix')
            
            if confusion_mat is None:
                logger.warning("No confusion matrix found in model results")
                return plots
            
            confusion_mat = np.array(confusion_mat)
            
            # Create confusion matrix plots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Raw confusion matrix
            ax1 = axes[0]
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Confusion Matrix (Counts)')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Add class labels if available
            application_type = hyperparameters.get('application_type', 'generic')
            if application_type == 'landslide_susceptibility':
                class_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High'][:confusion_mat.shape[0]]
            elif application_type == 'mineral_targeting':
                class_labels = ['Background', 'Low', 'Moderate', 'High'][:confusion_mat.shape[0]]
            else:
                class_labels = [f'Class {i}' for i in range(confusion_mat.shape[0])]
            
            ax1.set_xticklabels(class_labels, rotation=45)
            ax1.set_yticklabels(class_labels, rotation=0)
            
            # Plot 2: Normalized confusion matrix
            ax2 = axes[1]
            confusion_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
            sns.heatmap(confusion_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
            ax2.set_title('Normalized Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            ax2.set_xticklabels(class_labels, rotation=45)
            ax2.set_yticklabels(class_labels, rotation=0)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self._save_plot(fig, output_dir, 'confusion_matrix', hyperparameters)
            plots['confusion_matrix'] = plot_path
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create confusion matrix plots: {str(e)}")
        
        return plots
    
    def _create_roc_curve_plots(self, model_results: Dict[str, Any],
                              hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create ROC curve plots."""
        plots = {}
        output_dir = self._get_output_dir(context, 'roc_curve')
        
        try:
            metrics = model_results.get('model_metrics', {})
            roc_data = metrics.get('roc_curve')
            
            if roc_data is None:
                logger.warning("No ROC curve data found in model results")
                return plots
            
            # Extract ROC curve data
            fpr = np.array(roc_data['fpr'])
            tpr = np.array(roc_data['tpr'])
            auc_score = metrics.get('roc_auc', auc(fpr, tpr))
            
            # Create ROC curve plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add interpretation text
            interpretation = self._interpret_auc_score(auc_score)
            ax.text(0.6, 0.2, interpretation, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            # Save plot
            plot_path = self._save_plot(fig, output_dir, 'roc_curve', hyperparameters)
            plots['roc_curve'] = plot_path
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create ROC curve plots: {str(e)}")
        
        return plots
    
    def _create_precision_recall_plots(self, model_results: Dict[str, Any],
                                     hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create precision-recall curve plots."""
        plots = {}
        output_dir = self._get_output_dir(context, 'precision_recall')
        
        try:
            metrics = model_results.get('model_metrics', {})
            pr_data = metrics.get('precision_recall_curve')
            
            if pr_data is None:
                logger.warning("No precision-recall curve data found in model results")
                return plots
            
            # Extract PR curve data
            precision = np.array(pr_data['precision'])
            recall = np.array(pr_data['recall'])
            pr_auc = auc(recall, precision)
            
            # Create PR curve plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
            
            # Add baseline (random classifier)
            baseline = np.sum(model_results.get('y_true', [1])) / len(model_results.get('y_true', [1, 0]))
            ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                      label=f'Random Classifier (Precision = {baseline:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = self._save_plot(fig, output_dir, 'precision_recall_curve', hyperparameters)
            plots['precision_recall_curve'] = plot_path
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create precision-recall plots: {str(e)}")
        
        return plots
    
    def _create_residual_plots(self, model_results: Dict[str, Any],
                             hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create residual analysis plots for regression models."""
        plots = {}
        output_dir = self._get_output_dir(context, 'residuals')
        
        try:
            # Check if we have regression results
            y_true = model_results.get('y_true')
            y_pred = model_results.get('y_pred')
            
            if y_true is None or y_pred is None:
                logger.warning("No prediction data found for residual plots")
                return plots
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            residuals = y_true - y_pred
            
            # Create residual analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Residuals vs Predicted
            ax1 = axes[0, 0]
            ax1.scatter(y_pred, residuals, alpha=0.6)
            ax1.axhline(y=0, color='red', linestyle='--')
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Predicted')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Q-Q plot of residuals
            ax2 = axes[0, 1]
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot of Residuals')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Histogram of residuals
            ax3 = axes[1, 0]
            ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(residuals), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(residuals):.3f}')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Residuals')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Predicted vs Actual
            ax4 = axes[1, 1]
            ax4.scatter(y_true, y_pred, alpha=0.6)
            
            # Perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax4.set_xlabel('Actual Values')
            ax4.set_ylabel('Predicted Values')
            ax4.set_title('Predicted vs Actual')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Calculate and display R²
            r2 = r2_score(y_true, y_pred)
            ax4.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self._save_plot(fig, output_dir, 'residual_analysis', hyperparameters)
            plots['residual_analysis'] = plot_path
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create residual plots: {str(e)}")
        
        return plots
    
    def _create_distribution_plots(self, model_results: Dict[str, Any],
                                 hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create data distribution plots."""
        plots = {}
        output_dir = self._get_output_dir(context, 'distribution')
        
        try:
            # Get prediction probabilities or values
            probabilities = model_results.get('probabilities')
            predictions = model_results.get('predictions')
            
            if probabilities is None and predictions is None:
                logger.warning("No probability or prediction data found for distribution plots")
                return plots
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Prediction distribution
            ax1 = axes[0, 0]
            if predictions is not None:
                predictions = np.array(predictions)
                if len(np.unique(predictions)) < 10:  # Categorical
                    unique_vals, counts = np.unique(predictions, return_counts=True)
                    ax1.bar(unique_vals, counts, alpha=0.7)
                    ax1.set_xlabel('Prediction Class')
                else:  # Continuous
                    ax1.hist(predictions, bins=30, alpha=0.7, edgecolor='black')
                    ax1.set_xlabel('Prediction Value')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Prediction Distribution')
            
            # Plot 2: Probability distribution (if available)
            ax2 = axes[0, 1]
            if probabilities is not None:
                probabilities = np.array(probabilities)
                if probabilities.ndim > 1:
                    # Multi-class probabilities - show max probability distribution
                    max_probs = np.max(probabilities, axis=1)
                    ax2.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
                    ax2.set_xlabel('Maximum Probability')
                else:
                    # Binary probabilities
                    ax2.hist(probabilities, bins=30, alpha=0.7, edgecolor='black')
                    ax2.set_xlabel('Probability')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Probability Distribution')
            else:
                ax2.text(0.5, 0.5, 'No probability data available',
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Probability Distribution')
            
            # Plot 3: Class distribution (if classification)
            ax3 = axes[1, 0]
            y_true = model_results.get('y_true')
            if y_true is not None:
                y_true = np.array(y_true)
                unique_vals, counts = np.unique(y_true, return_counts=True)
                bars = ax3.bar(unique_vals, counts, alpha=0.7)
                ax3.set_xlabel('True Class')
                ax3.set_ylabel('Count')
                ax3.set_title('True Class Distribution')
                
                # Add percentage labels
                total = len(y_true)
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                            f'{count/total*100:.1f}%', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No true labels available',
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('True Class Distribution')
            
            # Plot 4: Confidence distribution
            ax4 = axes[1, 1]
            confidence_scores = model_results.get('confidence_scores')
            if confidence_scores is not None:
                confidence_scores = np.array(confidence_scores)
                ax4.hist(confidence_scores, bins=30, alpha=0.7, edgecolor='black')
                ax4.axvline(np.mean(confidence_scores), color='red', linestyle='--',
                           label=f'Mean: {np.mean(confidence_scores):.3f}')
                ax4.set_xlabel('Confidence Score')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Model Confidence Distribution')
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'No confidence data available',
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Model Confidence Distribution')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self._save_plot(fig, output_dir, 'distribution_analysis', hyperparameters)
            plots['distribution_analysis'] = plot_path
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create distribution plots: {str(e)}")
        
        return plots
    
    def _create_summary_dashboard(self, model_results: Dict[str, Any], generated_plots: Dict[str, Any],
                                hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create interactive summary dashboard using Plotly."""
        dashboard = {}
        output_dir = self._get_output_dir(context, 'dashboard')
        
        try:
            # Create interactive dashboard with Plotly
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Model Performance', 'Feature Importance', 
                              'Prediction Distribution', 'Confidence Analysis',
                              'Cross-Validation Results', 'Model Summary'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "box"}, {"type": "table"}]]
            )
            
            # Performance metrics
            metrics = model_results.get('model_metrics', {})
            if metrics:
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                metric_values = [metrics.get('accuracy', 0), metrics.get('precision', 0),
                               metrics.get('recall', 0), metrics.get('f1_score', 0)]
                
                fig.add_trace(
                    go.Bar(x=metric_names, y=metric_values, name='Performance',
                          marker_color='lightblue'),
                    row=1, col=1
                )
            
            # Feature importance
            feature_importance = model_results.get('feature_importance', {})
            if feature_importance and 'importance_scores' in feature_importance:
                importance = np.array(feature_importance['importance_scores'])
                feature_names = feature_importance.get('feature_names', [])
                
                # Top 10 features
                top_indices = np.argsort(importance)[-10:][::-1]
                top_importance = importance[top_indices]
                top_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                           for i in top_indices]
                
                fig.add_trace(
                    go.Bar(x=top_importance, y=top_names, name='Importance',
                          orientation='h', marker_color='lightcoral'),
                    row=1, col=2
                )
            
            # Prediction distribution
            predictions = model_results.get('predictions')
            if predictions is not None:
                fig.add_trace(
                    go.Histogram(x=predictions, name='Predictions', 
                               marker_color='lightgreen'),
                    row=2, col=1
                )
            
            # Confidence distribution
            confidence_scores = model_results.get('confidence_scores')
            if confidence_scores is not None:
                fig.add_trace(
                    go.Histogram(x=confidence_scores, name='Confidence',
                               marker_color='lightyellow'),
                    row=2, col=2
                )
            
            # Cross-validation results
            cv_results = model_results.get('cross_validation_results', {})
            if cv_results and 'cv_results' in cv_results:
                cv_data = cv_results['cv_results']
                for metric_name, metric_data in cv_data.items():
                    scores = metric_data.get('test_scores', [])
                    if scores:
                        fig.add_trace(
                            go.Box(y=scores, name=metric_name, boxmean=True),
                            row=3, col=1
                        )
            
            # Model summary table
            summary_data = self._create_model_summary_table(model_results, hyperparameters)
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                              fill_color='lightblue',
                              align='left'),
                    cells=dict(values=[list(summary_data.keys()), list(summary_data.values())],
                             fill_color='white',
                             align='left')
                ),
                row=3, col=2
            )
            
            # Update layout
            application_type = hyperparameters.get('application_type', 'generic')
            if application_type == 'landslide_susceptibility':
                title = 'Landslide Susceptibility Model Dashboard'
            elif application_type == 'mineral_targeting':
                title = 'Mineral Prospectivity Model Dashboard'
            else:
                title = 'Model Performance Dashboard'
            
            fig.update_layout(
                title=title,
                height=1200,
                showlegend=False,
                template='plotly_white'
            )
            
            # Save interactive dashboard
            dashboard_path = os.path.join(output_dir, 'model_dashboard.html')
            fig.write_html(dashboard_path)
            
            dashboard['interactive_dashboard'] = dashboard_path
            
            logger.info(f"Created interactive dashboard: {dashboard_path}")
            
        except Exception as e:
            logger.error(f"Failed to create summary dashboard: {str(e)}")
        
        return dashboard
    
    def _create_model_summary_table(self, model_results: Dict[str, Any], 
                                   hyperparameters: Dict[str, Any]) -> Dict[str, str]:
        """Create model summary statistics table."""
        summary = {}
        
        # Basic model info
        metrics = model_results.get('model_metrics', {})
        
        if 'accuracy' in metrics:
            summary['Accuracy'] = f"{metrics['accuracy']:.3f}"
        if 'roc_auc' in metrics:
            summary['AUC-ROC'] = f"{metrics['roc_auc']:.3f}"
        if 'f1_score' in metrics:
            summary['F1-Score'] = f"{metrics['f1_score']:.3f}"
        if 'r2_score' in metrics:
            summary['R² Score'] = f"{metrics['r2_score']:.3f}"
        
        # Cross-validation results
        cv_results = model_results.get('cross_validation_results', {})
        if cv_results and 'cv_summary' in cv_results:
            cv_summary = cv_results['cv_summary']
            primary_score = cv_summary.get('primary_score_mean', 0)
            summary['CV Score'] = f"{primary_score:.3f} ± {cv_summary.get('primary_score_std', 0):.3f}"
        
        # Feature count
        feature_importance = model_results.get('feature_importance', {})
        if 'feature_names' in feature_importance:
            summary['# Features'] = str(len(feature_importance['feature_names']))
        
        # Application type
        summary['Application'] = hyperparameters.get('application_type', 'Generic').replace('_', ' ').title()
        
        return summary
    
    def _group_features_by_type(self, feature_names: List[str], importance_scores: np.ndarray) -> Dict[str, List[float]]:
        """Group features by type based on naming patterns."""
        feature_groups = {
            'Spectral': [],
            'Topographic': [],
            'Texture': [],
            'Derived': [],
            'Other': []
        }
        
        for name, importance in zip(feature_names, importance_scores):
            name_lower = name.lower()
            
            if any(keyword in name_lower for keyword in ['band', 'ndvi', 'ndwi', 'spectral', 'reflectance']):
                feature_groups['Spectral'].append(importance)
            elif any(keyword in name_lower for keyword in ['elevation', 'slope', 'aspect', 'curvature', 'tpi']):
                feature_groups['Topographic'].append(importance)
            elif any(keyword in name_lower for keyword in ['texture', 'glcm', 'contrast', 'homogeneity']):
                feature_groups['Texture'].append(importance)
            elif any(keyword in name_lower for keyword in ['ratio', 'index', 'derived']):
                feature_groups['Derived'].append(importance)
            else:
                feature_groups['Other'].append(importance)
        
        # Remove empty groups
        feature_groups = {k: v for k, v in feature_groups.items() if v}
        
        return feature_groups
    
    def _interpret_auc_score(self, auc_score: float) -> str:
        """Provide interpretation of AUC score."""
        if auc_score >= 0.9:
            return "Excellent\nperformance"
        elif auc_score >= 0.8:
            return "Good\nperformance"
        elif auc_score >= 0.7:
            return "Fair\nperformance"
        elif auc_score >= 0.6:
            return "Poor\nperformance"
        else:
            return "Very poor\nperformance"
    
    def _get_output_dir(self, context: Dict[str, Any], plot_type: str) -> str:
        """Get output directory for specific plot type."""
        base_dir = context.get('output_dir', 'outputs/visualization')
        output_dir = os.path.join(base_dir, plot_type)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _save_plot(self, fig, output_dir: str, filename: str, 
                   hyperparameters: Dict[str, Any]) -> str:
        """Save plot in specified format(s)."""
        plot_format = hyperparameters.get('plot_format', 'png')
        dpi = hyperparameters.get('dpi', 300)
        
        if plot_format == 'both':
            # Save both PNG and PDF
            png_path = os.path.join(output_dir, f'{filename}.png')
            pdf_path = os.path.join(output_dir, f'{filename}.pdf')
            
            fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
            
            return png_path  # Return primary format
        else:
            # Save in single format
            file_path = os.path.join(output_dir, f'{filename}.{plot_format}')
            
            if plot_format == 'png':
                fig.savefig(file_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            else:
                fig.savefig(file_path, bbox_inches='tight', facecolor='white')
            
            return file_path
    
    def _generate_plot_metadata(self, generated_plots: Dict[str, Any], 
                              model_results: Dict[str, Any],
                              hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata about created plots."""
        metadata = {
            'plot_types': list(generated_plots.keys()),
            'total_plots': sum(len(plots) if isinstance(plots, dict) else 1 
                             for plots in generated_plots.values()),
            'plot_format': hyperparameters.get('plot_format', 'png'),
            'figure_size': hyperparameters.get('figure_size', [12, 8]),
            'application_type': hyperparameters.get('application_type', 'generic'),
            'model_type': model_results.get('model_type', 'unknown'),
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add plot-specific metadata
        plot_details = {}
        for plot_type, plots in generated_plots.items():
            if isinstance(plots, dict):
                plot_details[plot_type] = {
                    'n_plots': len(plots),
                    'plot_files': list(plots.values())
                }
            else:
                plot_details[plot_type] = {
                    'n_plots': 1,
                    'plot_files': [plots] if isinstance(plots, str) else []
                }
        
        metadata['plot_details'] = plot_details
        
        return metadata
    
    # Placeholder methods for additional plot types
    def _create_correlation_plots(self, model_results: Dict[str, Any],
                                hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create correlation analysis plots."""
        return {}
    
    def _create_validation_curve_plots(self, model_results: Dict[str, Any],
                                     hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create validation curve plots."""
        return {}
    
    def _create_learning_curve_plots(self, model_results: Dict[str, Any],
                                   hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create learning curve plots."""
        return {}
    
    def _create_model_comparison_plots(self, model_results: Dict[str, Any],
                                     hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Create model comparison plots."""
        return {}
    
    def get_plot_objects(self) -> Dict[str, Any]:
        """Get generated plot objects."""
        return self.plot_objects_
    
    def get_plot_metadata(self) -> Dict[str, Any]:
        """Get plot metadata."""
        return self.plot_metadata_
