"""
Automated report generation step for creating comprehensive analysis reports.

This step generates professional PDF and HTML reports summarizing landslide
susceptibility and mineral targeting analysis results with charts, maps, and
statistical summaries.
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template, Environment, FileSystemLoader
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

# Optional imports for PDF generation
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logging.warning("WeasyPrint not available. PDF generation will be limited.")

try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("ReportLab not available. PDF generation will use alternative methods.")

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class ReportGenerationStep(BaseStep):
    """
    Automated report generation step.
    
    Creates comprehensive analysis reports in HTML and PDF formats with
    executive summaries, detailed analysis, visualizations, and recommendations.
    """
    
    def __init__(self):
        super().__init__()
        self.report_data_: Dict[str, Any] = {}
        self.generated_reports_: Dict[str, str] = {}
        
    def get_step_type(self) -> str:
        return "report_generation"
    
    def get_required_inputs(self) -> list:
        return ['analysis_results']
    
    def get_outputs(self) -> list:
        return ['generated_reports', 'report_metadata']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate report generation hyperparameters."""
        # Validate report formats
        report_formats = hyperparameters.get('report_formats', ['html'])
        valid_formats = ['html', 'pdf', 'both']
        
        for fmt in report_formats:
            if fmt not in valid_formats:
                logger.error(f"Invalid report_format: {fmt}. Must be one of {valid_formats}")
                return False
        
        # Validate report sections
        report_sections = hyperparameters.get('report_sections', ['executive_summary', 'methodology', 'results'])
        valid_sections = [
            'executive_summary', 'methodology', 'data_sources', 'preprocessing',
            'feature_extraction', 'modeling', 'results', 'validation', 
            'recommendations', 'limitations', 'conclusions', 'appendix'
        ]
        
        for section in report_sections:
            if section not in valid_sections:
                logger.warning(f"Unknown report section: {section}")
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute report generation.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing generated reports and metadata
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
            
            # Load analysis results
            analysis_results = self._load_analysis_results(
                context['inputs']['analysis_results'], hyperparameters
            )
            
            if not analysis_results:
                return {
                    'status': 'failed',
                    'error': 'Failed to load analysis results',
                    'outputs': {}
                }
            
            # Prepare report data
            report_data = self._prepare_report_data(
                analysis_results, hyperparameters, context
            )
            
            # Generate requested report formats
            report_formats = hyperparameters.get('report_formats', ['html'])
            if 'both' in report_formats:
                report_formats = ['html', 'pdf']
            
            generated_reports = {}
            
            for report_format in report_formats:
                try:
                    if report_format == 'html':
                        html_report = self._generate_html_report(
                            report_data, hyperparameters, context
                        )
                        if html_report:
                            generated_reports['html'] = html_report
                    
                    elif report_format == 'pdf':
                        pdf_report = self._generate_pdf_report(
                            report_data, hyperparameters, context
                        )
                        if pdf_report:
                            generated_reports['pdf'] = pdf_report
                            
                except Exception as e:
                    logger.warning(f"Failed to generate {report_format} report: {str(e)}")
            
            # Generate report metadata
            report_metadata = self._generate_report_metadata(
                generated_reports, report_data, hyperparameters
            )
            
            # Store results
            self.report_data_ = report_data
            self.generated_reports_ = generated_reports
            
            # Prepare outputs
            outputs = {
                'generated_reports': generated_reports,
                'report_metadata': report_metadata
            }
            
            logger.info("Report generation completed successfully")
            logger.info(f"Generated {len(generated_reports)} report format(s)")
            
            return {
                'status': 'success',
                'message': 'Reports generated successfully',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'report_formats': list(generated_reports.keys()),
                    'application_type': hyperparameters.get('application_type', 'generic')
                }
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_analysis_results(self, analysis_input: Union[str, Dict[str, Any]], 
                             hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Load analysis results from various sources."""
        try:
            if isinstance(analysis_input, dict):
                return analysis_input
            elif isinstance(analysis_input, str):
                # Load from file
                if analysis_input.endswith('.json'):
                    import json
                    with open(analysis_input, 'r') as f:
                        return json.load(f)
                elif analysis_input.endswith('.pkl'):
                    import joblib
                    return joblib.load(analysis_input)
            
            logger.error(f"Unsupported analysis results format: {type(analysis_input)}")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to load analysis results: {str(e)}")
            return {}
    
    def _prepare_report_data(self, analysis_results: Dict[str, Any], 
                           hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and structure data for report generation."""
        application_type = hyperparameters.get('application_type', 'generic')
        
        report_data = {
            'metadata': {
                'title': self._get_report_title(application_type),
                'subtitle': hyperparameters.get('report_subtitle', ''),
                'author': hyperparameters.get('author', 'Automated Analysis System'),
                'organization': hyperparameters.get('organization', ''),
                'date': datetime.now().strftime('%B %d, %Y'),
                'application_type': application_type,
                'study_area': hyperparameters.get('study_area', 'Study Area'),
                'version': hyperparameters.get('version', '1.0')
            },
            'executive_summary': self._create_executive_summary(analysis_results, application_type),
            'methodology': self._create_methodology_section(analysis_results, hyperparameters),
            'data_sources': self._create_data_sources_section(analysis_results),
            'results': self._create_results_section(analysis_results, application_type),
            'validation': self._create_validation_section(analysis_results),
            'recommendations': self._create_recommendations_section(analysis_results, application_type),
            'conclusions': self._create_conclusions_section(analysis_results, application_type),
            'visualizations': self._prepare_visualizations(analysis_results, context),
            'statistics': self._prepare_statistics_tables(analysis_results)
        }
        
        return report_data
    
    def _get_report_title(self, application_type: str) -> str:
        """Get appropriate report title based on application type."""
        if application_type == 'landslide_susceptibility':
            return 'Landslide Susceptibility Assessment Report'
        elif application_type == 'mineral_targeting':
            return 'Mineral Prospectivity Analysis Report'
        else:
            return 'Geospatial Analysis Report'
    
    def _create_executive_summary(self, analysis_results: Dict[str, Any], application_type: str) -> Dict[str, Any]:
        """Create executive summary section."""
        summary = {
            'objectives': self._get_objectives_text(application_type),
            'key_findings': [],
            'recommendations_summary': [],
            'conclusions': []
        }
        
        # Extract key findings from results
        model_metrics = analysis_results.get('model_metrics', {})
        
        if application_type == 'landslide_susceptibility':
            if 'accuracy' in model_metrics:
                accuracy = model_metrics['accuracy']
                summary['key_findings'].append(
                    f"The landslide susceptibility model achieved {accuracy:.1%} accuracy in predicting landslide occurrence."
                )
            
            if 'roc_auc' in model_metrics:
                auc = model_metrics['roc_auc']
                performance_level = self._interpret_model_performance(auc)
                summary['key_findings'].append(
                    f"Model performance is rated as '{performance_level}' with an AUC score of {auc:.3f}."
                )
            
            # Risk distribution
            prediction_stats = analysis_results.get('prediction_statistics', {})
            if 'class_distribution' in prediction_stats:
                class_dist = prediction_stats['class_distribution']
                high_risk_areas = sum(v.get('percentage', 0) for k, v in class_dist.items() 
                                    if 'high' in k.lower())
                summary['key_findings'].append(
                    f"Approximately {high_risk_areas:.1f}% of the study area is classified as high to very high susceptibility."
                )
        
        elif application_type == 'mineral_targeting':
            if 'accuracy' in model_metrics:
                accuracy = model_metrics['accuracy']
                summary['key_findings'].append(
                    f"The mineral prospectivity model achieved {accuracy:.1%} accuracy in identifying potential areas."
                )
            
            # Prospectivity distribution
            prediction_stats = analysis_results.get('prediction_statistics', {})
            if 'class_distribution' in prediction_stats:
                class_dist = prediction_stats['class_distribution']
                high_potential = sum(v.get('percentage', 0) for k, v in class_dist.items() 
                                   if 'high' in k.lower() or 'moderate' in k.lower())
                summary['key_findings'].append(
                    f"Approximately {high_potential:.1f}% of the study area shows moderate to high mineral potential."
                )
        
        # Add general findings
        feature_importance = analysis_results.get('feature_importance', {})
        if 'top_features' in feature_importance:
            top_features = feature_importance['top_features'][:3]
            feature_names = [f['feature'] for f in top_features]
            summary['key_findings'].append(
                f"The most important predictive features are: {', '.join(feature_names)}."
            )
        
        return summary
    
    def _get_objectives_text(self, application_type: str) -> str:
        """Get objectives text based on application type."""
        if application_type == 'landslide_susceptibility':
            return ("This study aims to assess landslide susceptibility across the study area using "
                   "machine learning techniques and remote sensing data to identify areas at risk "
                   "of landslide occurrence.")
        elif application_type == 'mineral_targeting':
            return ("This analysis focuses on identifying areas with high mineral prospectivity "
                   "using integrated geospatial datasets and advanced analytical techniques to "
                   "guide exploration efforts.")
        else:
            return ("This report presents the results of a comprehensive geospatial analysis "
                   "aimed at identifying patterns and relationships in the study area.")
    
    def _create_methodology_section(self, analysis_results: Dict[str, Any], 
                                  hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create methodology section."""
        methodology = {
            'overview': '',
            'data_processing': [],
            'feature_extraction': [],
            'modeling_approach': [],
            'validation_strategy': []
        }
        
        application_type = hyperparameters.get('application_type', 'generic')
        
        if application_type == 'landslide_susceptibility':
            methodology['overview'] = (
                "The landslide susceptibility assessment follows a systematic approach combining "
                "remote sensing data, digital elevation models, and machine learning techniques. "
                "The methodology includes data preprocessing, feature extraction, model training, "
                "and validation using established protocols."
            )
        elif application_type == 'mineral_targeting':
            methodology['overview'] = (
                "The mineral prospectivity analysis employs a data-driven approach integrating "
                "multispectral imagery, geological data, and advanced analytics. The workflow "
                "encompasses spectral analysis, feature engineering, predictive modeling, and "
                "uncertainty assessment."
            )
        
        # Add processing steps based on available results
        if 'preprocessing_results' in analysis_results:
            methodology['data_processing'].append("Atmospheric correction and radiometric calibration")
            methodology['data_processing'].append("Geometric correction and spatial harmonization")
        
        if 'feature_extraction_results' in analysis_results:
            methodology['feature_extraction'].append("Spectral indices calculation")
            methodology['feature_extraction'].append("Topographic feature derivation")
            methodology['feature_extraction'].append("Texture analysis")
        
        # Model information
        model_info = analysis_results.get('model_info', {})
        if model_info:
            model_type = model_info.get('model_type', 'Machine Learning')
            methodology['modeling_approach'].append(f"{model_type} algorithm implementation")
            methodology['modeling_approach'].append("Hyperparameter optimization")
            methodology['modeling_approach'].append("Cross-validation training")
        
        return methodology
    
    def _create_data_sources_section(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create data sources section."""
        data_sources = {
            'primary_datasets': [],
            'auxiliary_data': [],
            'temporal_coverage': '',
            'spatial_resolution': '',
            'data_quality': ''
        }
        
        # Extract data source information from results
        data_info = analysis_results.get('data_sources', {})
        
        if 'satellite_data' in data_info:
            data_sources['primary_datasets'].append("Sentinel-2 multispectral imagery")
        if 'dem_data' in data_info:
            data_sources['primary_datasets'].append("Digital Elevation Model (DEM)")
        if 'inventory_data' in data_info:
            data_sources['auxiliary_data'].append("Historical inventory data")
        
        return data_sources
    
    def _create_results_section(self, analysis_results: Dict[str, Any], application_type: str) -> Dict[str, Any]:
        """Create results section."""
        results = {
            'model_performance': {},
            'spatial_distribution': {},
            'feature_analysis': {},
            'uncertainty_assessment': {}
        }
        
        # Model performance
        model_metrics = analysis_results.get('model_metrics', {})
        results['model_performance'] = {
            'accuracy_metrics': model_metrics,
            'performance_interpretation': self._interpret_model_performance(
                model_metrics.get('roc_auc', model_metrics.get('accuracy', 0))
            )
        }
        
        # Spatial distribution
        prediction_stats = analysis_results.get('prediction_statistics', {})
        results['spatial_distribution'] = prediction_stats
        
        # Feature analysis
        feature_importance = analysis_results.get('feature_importance', {})
        results['feature_analysis'] = feature_importance
        
        # Uncertainty assessment
        uncertainty_results = analysis_results.get('uncertainty_results', {})
        results['uncertainty_assessment'] = uncertainty_results
        
        return results
    
    def _create_validation_section(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation section."""
        validation = {
            'cross_validation': {},
            'statistical_tests': {},
            'spatial_validation': {},
            'validation_interpretation': ''
        }
        
        cv_results = analysis_results.get('cross_validation_results', {})
        validation['cross_validation'] = cv_results
        
        statistical_tests = analysis_results.get('statistical_tests', {})
        validation['statistical_tests'] = statistical_tests
        
        # Interpretation
        if cv_results and 'cv_summary' in cv_results:
            cv_score = cv_results['cv_summary'].get('primary_score_mean', 0)
            validation['validation_interpretation'] = (
                f"Cross-validation results indicate {self._interpret_model_performance(cv_score)} "
                f"model performance with a mean score of {cv_score:.3f}."
            )
        
        return validation
    
    def _create_recommendations_section(self, analysis_results: Dict[str, Any], application_type: str) -> List[str]:
        """Create recommendations based on results."""
        recommendations = []
        
        model_metrics = analysis_results.get('model_metrics', {})
        
        if application_type == 'landslide_susceptibility':
            recommendations.extend([
                "Focus monitoring and mitigation efforts on areas classified as high to very high susceptibility.",
                "Implement early warning systems in populated high-risk zones.",
                "Conduct detailed geotechnical investigations in priority areas.",
                "Update susceptibility maps regularly with new landslide inventory data."
            ])
            
            # Performance-based recommendations
            accuracy = model_metrics.get('accuracy', 0)
            if accuracy < 0.8:
                recommendations.append(
                    "Consider incorporating additional predictor variables to improve model accuracy."
                )
        
        elif application_type == 'mineral_targeting':
            recommendations.extend([
                "Prioritize exploration activities in areas with high mineral prospectivity.",
                "Conduct detailed geological surveys in moderate to high potential zones.",
                "Integrate geochemical and geophysical data to refine target areas.",
                "Validate prospectivity maps with ground-truth exploration data."
            ])
        
        # General recommendations
        feature_importance = analysis_results.get('feature_importance', {})
        if 'top_features' in feature_importance:
            top_feature = feature_importance['top_features'][0]['feature']
            recommendations.append(
                f"Pay special attention to {top_feature} as it shows the highest predictive importance."
            )
        
        return recommendations
    
    def _create_conclusions_section(self, analysis_results: Dict[str, Any], application_type: str) -> List[str]:
        """Create conclusions based on analysis results."""
        conclusions = []
        
        model_metrics = analysis_results.get('model_metrics', {})
        accuracy = model_metrics.get('accuracy', 0)
        auc = model_metrics.get('roc_auc', 0)
        
        if application_type == 'landslide_susceptibility':
            conclusions.extend([
                f"The developed landslide susceptibility model demonstrates {self._interpret_model_performance(max(accuracy, auc))} "
                f"performance in predicting landslide-prone areas.",
                "The integration of remote sensing and topographic data provides effective spatial prediction capability.",
                "The results can support risk management and land-use planning decisions."
            ])
        elif application_type == 'mineral_targeting':
            conclusions.extend([
                f"The mineral prospectivity analysis successfully identifies {self._interpret_model_performance(max(accuracy, auc))} "
                f"target areas for exploration.",
                "The methodology effectively integrates multi-source geospatial data for mineral exploration.",
                "The results provide valuable guidance for prioritizing exploration investments."
            ])
        
        return conclusions
    
    def _prepare_visualizations(self, analysis_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Prepare visualizations for inclusion in the report."""
        visualizations = {}
        
        # Look for existing plot files
        visualization_results = analysis_results.get('visualization_results', {})
        
        if 'statistical_plots' in visualization_results:
            plots = visualization_results['statistical_plots']
            for plot_type, plot_files in plots.items():
                if isinstance(plot_files, dict):
                    for plot_name, plot_path in plot_files.items():
                        if os.path.exists(plot_path):
                            visualizations[f"{plot_type}_{plot_name}"] = self._encode_image_base64(plot_path)
                elif isinstance(plot_files, str) and os.path.exists(plot_files):
                    visualizations[plot_type] = self._encode_image_base64(plot_files)
        
        # Create additional visualizations if needed
        if not visualizations:
            # Create basic performance chart
            performance_chart = self._create_performance_chart(analysis_results)
            if performance_chart:
                visualizations['performance_overview'] = performance_chart
        
        return visualizations
    
    def _prepare_statistics_tables(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare statistics tables for the report."""
        tables = {}
        
        # Model performance table
        model_metrics = analysis_results.get('model_metrics', {})
        if model_metrics:
            tables['model_performance'] = self._format_metrics_table(model_metrics)
        
        # Feature importance table
        feature_importance = analysis_results.get('feature_importance', {})
        if 'top_features' in feature_importance:
            tables['feature_importance'] = self._format_feature_importance_table(
                feature_importance['top_features'][:10]
            )
        
        # Class distribution table
        prediction_stats = analysis_results.get('prediction_statistics', {})
        if 'class_distribution' in prediction_stats:
            tables['class_distribution'] = self._format_class_distribution_table(
                prediction_stats['class_distribution']
            )
        
        return tables
    
    def _generate_html_report(self, report_data: Dict[str, Any], 
                            hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate HTML report."""
        try:
            output_dir = context.get('output_dir', 'outputs/reports')
            os.makedirs(output_dir, exist_ok=True)
            
            # Create HTML template
            html_template = self._get_html_template()
            
            # Render template with data
            template = Template(html_template)
            html_content = template.render(**report_data)
            
            # Save HTML file
            report_filename = hyperparameters.get('report_filename', 'analysis_report.html')
            html_path = os.path.join(output_dir, report_filename)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")
            return None
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], 
                           hyperparameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate PDF report."""
        try:
            output_dir = context.get('output_dir', 'outputs/reports')
            os.makedirs(output_dir, exist_ok=True)
            
            report_filename = hyperparameters.get('pdf_filename', 'analysis_report.pdf')
            pdf_path = os.path.join(output_dir, report_filename)
            
            if WEASYPRINT_AVAILABLE:
                # Use WeasyPrint for HTML to PDF conversion
                html_content = self._get_html_template()
                template = Template(html_content)
                rendered_html = template.render(**report_data)
                
                HTML(string=rendered_html).write_pdf(pdf_path)
                
            elif REPORTLAB_AVAILABLE:
                # Use ReportLab for direct PDF generation
                self._generate_reportlab_pdf(report_data, pdf_path)
                
            else:
                logger.warning("No PDF generation library available")
                return None
            
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {str(e)}")
            return None
    
    def _generate_reportlab_pdf(self, report_data: Dict[str, Any], pdf_path: str):
        """Generate PDF using ReportLab."""
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=1,  # Center alignment
            spaceAfter=30
        )
        
        story.append(Paragraph(report_data['metadata']['title'], title_style))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        executive_summary = report_data.get('executive_summary', {})
        objectives = executive_summary.get('objectives', '')
        if objectives:
            story.append(Paragraph(objectives, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Key Findings
        key_findings = executive_summary.get('key_findings', [])
        if key_findings:
            story.append(Paragraph("Key Findings:", styles['Heading4']))
            for finding in key_findings:
                story.append(Paragraph(f"• {finding}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Model Performance Table
        statistics = report_data.get('statistics', {})
        if 'model_performance' in statistics:
            story.append(Paragraph("Model Performance", styles['Heading3']))
            
            perf_data = statistics['model_performance']
            table_data = [['Metric', 'Value']]
            for metric, value in perf_data.items():
                table_data.append([metric.replace('_', ' ').title(), str(value)])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
    
    def _get_html_template(self) -> str:
        """Get HTML template for report generation."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header .subtitle {
            color: #7f8c8d;
            font-size: 1.2em;
        }
        .metadata {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .metadata table {
            width: 100%;
            border-collapse: collapse;
        }
        .metadata td {
            padding: 5px 10px;
            border: none;
        }
        .metadata .label {
            font-weight: bold;
            color: #2c3e50;
            width: 120px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .section h3 {
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        .key-findings {
            background-color: #e8f6f3;
            padding: 20px;
            border-left: 4px solid #27ae60;
            margin-bottom: 20px;
        }
        .key-findings ul {
            margin: 0;
            padding-left: 20px;
        }
        .key-findings li {
            margin-bottom: 10px;
        }
        .recommendations {
            background-color: #fef9e7;
            padding: 20px;
            border-left: 4px solid #f39c12;
            margin-bottom: 20px;
        }
        .recommendations ul {
            margin: 0;
            padding-left: 20px;
        }
        .recommendations li {
            margin-bottom: 10px;
        }
        .statistics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .statistics-table th,
        .statistics-table td {
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: left;
        }
        .statistics-table th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        .statistics-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .statistics-table tr:hover {
            background-color: #e3f2fd;
        }
        .visualization {
            text-align: center;
            margin: 30px 0;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .methodology ul,
        .conclusions ul {
            padding-left: 20px;
        }
        .methodology li,
        .conclusions li {
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{{ metadata.title }}</h1>
            {% if metadata.subtitle %}
            <div class="subtitle">{{ metadata.subtitle }}</div>
            {% endif %}
        </div>

        <!-- Metadata -->
        <div class="metadata">
            <table>
                <tr>
                    <td class="label">Study Area:</td>
                    <td>{{ metadata.study_area }}</td>
                    <td class="label">Date:</td>
                    <td>{{ metadata.date }}</td>
                </tr>
                <tr>
                    <td class="label">Author:</td>
                    <td>{{ metadata.author }}</td>
                    <td class="label">Version:</td>
                    <td>{{ metadata.version }}</td>
                </tr>
                {% if metadata.organization %}
                <tr>
                    <td class="label">Organization:</td>
                    <td colspan="3">{{ metadata.organization }}</td>
                </tr>
                {% endif %}
            </table>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <p>{{ executive_summary.objectives }}</p>
            
            {% if executive_summary.key_findings %}
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                {% for finding in executive_summary.key_findings %}
                    <li>{{ finding }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>

        <!-- Methodology -->
        {% if methodology.overview %}
        <div class="section">
            <h2>Methodology</h2>
            <p>{{ methodology.overview }}</p>
            
            {% if methodology.data_processing %}
            <h3>Data Processing</h3>
            <ul>
            {% for step in methodology.data_processing %}
                <li>{{ step }}</li>
            {% endfor %}
            </ul>
            {% endif %}
            
            {% if methodology.modeling_approach %}
            <h3>Modeling Approach</h3>
            <ul>
            {% for step in methodology.modeling_approach %}
                <li>{{ step }}</li>
            {% endfor %}
            </ul>
            {% endif %}
        </div>
        {% endif %}

        <!-- Results -->
        <div class="section">
            <h2>Results</h2>
            
            <!-- Model Performance Table -->
            {% if statistics.model_performance %}
            <h3>Model Performance</h3>
            <table class="statistics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                {% for metric, value in statistics.model_performance.items() %}
                    <tr>
                        <td>{{ metric.replace('_', ' ').title() }}</td>
                        <td>{{ "%.3f"|format(value) if value is number else value }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            {% endif %}

            <!-- Feature Importance Table -->
            {% if statistics.feature_importance %}
            <h3>Top Predictive Features</h3>
            <table class="statistics-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
                </thead>
                <tbody>
                {% for item in statistics.feature_importance %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ item.feature }}</td>
                        <td>{{ "%.4f"|format(item.importance) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            {% endif %}

            <!-- Visualizations -->
            {% for viz_name, viz_data in visualizations.items() %}
            <div class="visualization">
                <h3>{{ viz_name.replace('_', ' ').title() }}</h3>
                <img src="data:image/png;base64,{{ viz_data }}" alt="{{ viz_name }}">
            </div>
            {% endfor %}
        </div>

        <!-- Recommendations -->
        {% if recommendations %}
        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>
                {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}

        <!-- Conclusions -->
        {% if conclusions %}
        <div class="section">
            <h2>Conclusions</h2>
            <div class="conclusions">
                <ul>
                {% for conclusion in conclusions %}
                    <li>{{ conclusion }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}

        <!-- Footer -->
        <div class="footer">
            <p>Generated automatically by the Modular Pipeline Orchestrator</p>
            <p>Report generated on {{ metadata.date }}</p>
        </div>
    </div>
</body>
</html>
        """
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 for embedding in HTML."""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            return ""
    
    def _create_performance_chart(self, analysis_results: Dict[str, Any]) -> Optional[str]:
        """Create a basic performance chart if no visualizations exist."""
        try:
            model_metrics = analysis_results.get('model_metrics', {})
            if not model_metrics:
                return None
            
            # Create a simple performance bar chart
            metrics = []
            values = []
            
            if 'accuracy' in model_metrics:
                metrics.append('Accuracy')
                values.append(model_metrics['accuracy'])
            if 'precision' in model_metrics:
                metrics.append('Precision')
                values.append(model_metrics['precision'])
            if 'recall' in model_metrics:
                metrics.append('Recall')
                values.append(model_metrics['recall'])
            if 'f1_score' in model_metrics:
                metrics.append('F1-Score')
                values.append(model_metrics['f1_score'])
            
            if not metrics:
                return None
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics, values, alpha=0.7, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
            plt.ylim(0, 1)
            plt.ylabel('Score')
            plt.title('Model Performance Metrics')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # Save to BytesIO and encode
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close()
            buffer.close()
            
            return encoded_string
            
        except Exception as e:
            logger.error(f"Failed to create performance chart: {str(e)}")
            return None
    
    def _format_metrics_table(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format metrics for table display."""
        formatted_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 0 <= value <= 1:
                    formatted_metrics[key] = f"{value:.3f}"
                else:
                    formatted_metrics[key] = f"{value:.2f}"
            else:
                formatted_metrics[key] = str(value)
        
        return formatted_metrics
    
    def _format_feature_importance_table(self, feature_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format feature importance for table display."""
        formatted_features = []
        
        for i, feature in enumerate(feature_list):
            formatted_features.append({
                'rank': i + 1,
                'feature': feature.get('feature', 'Unknown'),
                'importance': f"{feature.get('importance', 0):.4f}"
            })
        
        return formatted_features
    
    def _format_class_distribution_table(self, class_dist: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format class distribution for table display."""
        formatted_dist = []
        
        for class_name, stats in class_dist.items():
            formatted_dist.append({
                'class': class_name,
                'count': stats.get('count', 0),
                'percentage': f"{stats.get('percentage', 0):.1f}%"
            })
        
        return formatted_dist
    
    def _interpret_model_performance(self, score: float) -> str:
        """Interpret model performance score."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "fair"
        elif score >= 0.6:
            return "poor"
        else:
            return "very poor"
    
    def _generate_report_metadata(self, generated_reports: Dict[str, str], 
                                 report_data: Dict[str, Any],
                                 hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata about the created reports."""
        metadata = {
            'generated_formats': list(generated_reports.keys()),
            'report_files': generated_reports,
            'generation_timestamp': datetime.now().isoformat(),
            'application_type': hyperparameters.get('application_type', 'generic'),
            'report_sections': list(report_data.keys()),
            'total_visualizations': len(report_data.get('visualizations', {})),
            'total_tables': len(report_data.get('statistics', {}))
        }
        
        # Add file sizes
        file_sizes = {}
        for format_type, file_path in generated_reports.items():
            try:
                if os.path.exists(file_path):
                    size_bytes = os.path.getsize(file_path)
                    size_mb = size_bytes / (1024 * 1024)
                    file_sizes[format_type] = f"{size_mb:.2f} MB"
            except Exception:
                file_sizes[format_type] = "Unknown"
        
        metadata['file_sizes'] = file_sizes
        
        return metadata
    
    def get_report_data(self) -> Dict[str, Any]:
        """Get structured report data."""
        return self.report_data_
    
    def get_generated_reports(self) -> Dict[str, str]:
        """Get paths to generated reports."""
        return self.generated_reports_
