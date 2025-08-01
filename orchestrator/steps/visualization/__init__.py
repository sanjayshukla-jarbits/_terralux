"""
Visualization module for the Modular Pipeline Orchestrator.

This module contains visualization and reporting steps for creating compelling
visual outputs from landslide susceptibility and mineral targeting analyses.

Available visualization steps:
- MapVisualizationStep: Interactive web maps using Folium with multiple layers
- StatisticalPlotsStep: Statistical charts, performance metrics, and model comparisons
- ReportGenerationStep: Automated PDF/HTML report generation with analysis summaries

Each step is designed to handle both landslide susceptibility and mineral targeting
applications with domain-specific styling, legends, and interpretation frameworks.
"""

from .map_visualization_step import MapVisualizationStep
from .statistical_plots_step import StatisticalPlotsStep
from .report_generation_step import ReportGenerationStep

__all__ = [
    'MapVisualizationStep',
    'StatisticalPlotsStep',
    'ReportGenerationStep'
]

# Register all visualization steps with the registry
try:
    from ..base import StepRegistry, register_step_safe, BaseStep
    logger.debug("✓ StepRegistry imported successfully")
    REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"StepRegistry not available: {e}")
    StepRegistry = None
    register_step_safe = None
    BaseStep = None
    REGISTRY_AVAILABLE = False

def register_visualization_steps():
    """Register all visualization steps with the step registry."""
    # Map visualization aliases
    StepRegistry.register('map_visualization', MapVisualizationStep)
    StepRegistry.register('interactive_map', MapVisualizationStep)
    StepRegistry.register('folium_map', MapVisualizationStep)
    StepRegistry.register('web_map', MapVisualizationStep)
    StepRegistry.register('risk_map_visualization', MapVisualizationStep)
    StepRegistry.register('susceptibility_map', MapVisualizationStep)
    StepRegistry.register('prospectivity_map', MapVisualizationStep)
    
    # Statistical plots aliases
    StepRegistry.register('statistical_plots', StatisticalPlotsStep)
    StepRegistry.register('performance_plots', StatisticalPlotsStep)
    StepRegistry.register('model_comparison_plots', StatisticalPlotsStep)
    StepRegistry.register('feature_importance_plots', StatisticalPlotsStep)
    StepRegistry.register('validation_plots', StatisticalPlotsStep)
    
    # Report generation aliases  
    StepRegistry.register('report_generation', ReportGenerationStep)
    StepRegistry.register('html_report', ReportGenerationStep)
    StepRegistry.register('pdf_report', ReportGenerationStep)
    StepRegistry.register('analysis_report', ReportGenerationStep)
    StepRegistry.register('assessment_report', ReportGenerationStep)

# Auto-register when module is imported
register_visualization_steps()
