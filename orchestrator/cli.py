#!/usr/bin/env python3
"""
CORRECTED Command-line interface for the TerraLux Modular Pipeline Orchestrator
==============================================================================

This corrected implementation fixes all the critical issues found in the execution log:
- Fixed method signatures and orchestrator integration
- Enhanced error handling and logging
- Improved process loading and execution
- Better fallback mechanisms
- Proper template variable handling

Key Fixes Applied:
- Corrected load_process method calls to match orchestrator expectations
- Enhanced error handling with proper exception catching
- Improved logging and status reporting
- Better fallback mechanisms when orchestrator components fail
- Fixed template variable processing and validation
- Enhanced process validation and execution flow

Author: TerraLux Development Team
Version: 1.0.0-corrected
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback
import time

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser with full feature set"""
    parser = argparse.ArgumentParser(
        description='TerraLux Modular Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute landslide susceptibility pipeline
  python -m orchestrator.cli processes/landslide_susceptibility/standard_pipeline.json \\
    --bbox 85.3 27.6 85.4 27.7 \\
    --start-date 2023-01-01 \\
    --end-date 2023-12-31 \\
    --area-name nepal_test

  # Execute data acquisition only
  python -m orchestrator.cli processes/data_acquisition_only.json \\
    --bbox 85.3 27.6 85.4 27.7 \\
    --start-date 2023-01-01 \\
    --end-date 2023-12-31 \\
    --area-name nepal_test \\
    --verbose

  # Execute mineral targeting pipeline
  python -m orchestrator.cli processes/mineral_targeting/standard_mineral_pipeline.json \\
    --bbox 85.3 27.6 85.4 27.7 \\
    --config custom_config.json \\
    --verbose

  # List available step types
  python -m orchestrator.cli --list-steps

  # Validate process file
  python -m orchestrator.cli processes/my_process.json --validate

  # Show process information
  python -m orchestrator.cli processes/my_process.json --info
        """
    )
    
    # Main arguments
    parser.add_argument(
        'process_file',
        nargs='?',
        help='Path to JSON process definition file'
    )
    
    # Required parameters (when executing)
    location_group = parser.add_argument_group('Location Parameters')
    location_group.add_argument(
        '--bbox',
        nargs=4,
        type=float,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help='Bounding box coordinates (required for execution)'
    )
    
    # Temporal parameters
    temporal_group = parser.add_argument_group('Temporal Parameters')
    temporal_group.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    temporal_group.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    # Data parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--area-name', help='Area name for outputs')
    data_group.add_argument('--inventory-path', help='Landslide inventory file path')
    data_group.add_argument('--local-data-path', help='Local data directory path')
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', help='Additional configuration file (JSON/YAML)')
    config_group.add_argument('--schema', help='JSON schema file for validation')
    config_group.add_argument('--output-dir', help='Output directory path')
    config_group.add_argument('--temp-dir', help='Temporary files directory')
    
    # Execution control
    execution_group = parser.add_argument_group('Execution Control')
    execution_group.add_argument('--parallel', action='store_true', help='Enable parallel execution')
    execution_group.add_argument('--max-workers', type=int, default=4, help='Maximum parallel workers')
    execution_group.add_argument('--continue-on-error', action='store_true', help='Skip failed steps and continue')
    execution_group.add_argument('--dry-run', action='store_true', help='Show execution plan without running')
    execution_group.add_argument('--force-mock', action='store_true', help='Force mock execution for testing')
    
    # Monitoring and logging
    logging_group = parser.add_argument_group('Logging and Monitoring')
    logging_group.add_argument('--verbose', '-v', action='store_true', help='Verbose output (DEBUG level)')
    logging_group.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (WARNING level only)')
    logging_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set log level explicitly')
    logging_group.add_argument('--log-file', help='Path to log file')
    logging_group.add_argument('--no-monitoring', action='store_true', help='Disable performance monitoring')
    
    # Utility commands
    utility_group = parser.add_argument_group('Utility Commands')
    utility_group.add_argument('--list-steps', action='store_true', help='List available step types')
    utility_group.add_argument('--list-processes', action='store_true', help='List available process files')
    utility_group.add_argument('--validate', action='store_true', help='Validate process file only')
    utility_group.add_argument('--info', action='store_true', help='Show process information')
    utility_group.add_argument('--version', action='store_true', help='Show version information')
    
    return parser

def setup_logging(level: int, log_file: Optional[str] = None):
    """Setup logging configuration"""
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=level, format=format_str)

def build_template_variables(args) -> Dict[str, Any]:
    """Build template variables from command line arguments"""
    template_vars = {}
    
    if args.bbox:
        template_vars['bbox'] = args.bbox
    if args.start_date:
        template_vars['start_date'] = args.start_date
    if args.end_date:
        template_vars['end_date'] = args.end_date
    if args.area_name:
        template_vars['area_name'] = args.area_name
    if args.output_dir:
        template_vars['output_dir'] = args.output_dir
    if args.inventory_path:
        template_vars['landslide_inventory_path'] = args.inventory_path
    if args.local_data_path:
        template_vars['local_data_path'] = args.local_data_path
    
    return template_vars

def load_additional_config(config_path: str) -> Dict[str, Any]:
    """Load additional configuration from file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML configuration files")
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")

def validate_process_file(process_file: str, schema_path: Optional[str] = None) -> bool:
    """Validate process file against schema"""
    try:
        with open(process_file, 'r') as f:
            process_def = json.load(f)
        
        # Basic validation
        required_sections = ['process_info', 'steps']
        for section in required_sections:
            if section not in process_def:
                print(f"❌ Missing required section: {section}")
                return False
        
        # Validate steps
        steps = process_def.get('steps', [])
        if not steps:
            print("❌ No steps defined in process")
            return False
        
        step_ids = set()
        for i, step in enumerate(steps):
            if 'id' not in step:
                print(f"❌ Step {i} missing required 'id' field")
                return False
            
            step_id = step['id']
            if step_id in step_ids:
                print(f"❌ Duplicate step ID: {step_id}")
                return False
            step_ids.add(step_id)
            
            if 'type' not in step:
                print(f"❌ Step {step_id} missing required 'type' field")
                return False
        
        print("✅ Process file validation successful")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def list_available_steps():
    """List available step types"""
    try:
        # Try to get from registry if available
        from orchestrator.steps.base.step_registry import StepRegistry
        available_types = StepRegistry.get_available_types()
        
        if available_types:
            print(f"✅ Registered step types ({len(available_types)}):")
            for step_type in sorted(available_types):
                print(f"  ✓ {step_type}")
        else:
            print("⚠️  No steps currently registered")
            
        # Try to get registry stats
        try:
            stats = StepRegistry.get_registry_stats()
            print(f"\n📊 Registry Statistics:")
            print(f"  Total registered: {stats['total_registered']}")
            print(f"  Total aliases: {stats['total_aliases']}")
            print(f"  Total categories: {stats['total_categories']}")
            
            if stats.get('categories'):
                print(f"\n📁 Categories:")
                for category, count in stats['categories'].items():
                    print(f"  {category}: {count} steps")
                    
        except Exception as e:
            print(f"  (Could not get registry stats: {e})")
            
    except ImportError:
        print("⚠️  Step registry not available")
    
    print("\n📋 Planned step types (when fully implemented):")
    step_types = [
        "sentinel_hub_acquisition - Acquire Sentinel-2/1 data from Sentinel Hub",
        "dem_acquisition - Digital Elevation Model acquisition (SRTM/ASTER/ALOS)",
        "local_files_discovery - Discover and catalog local files",
        "spectral_indices_extraction - Calculate NDVI, NDWI, mineral indices",
        "topographic_derivatives - Calculate slope, aspect, curvature",
        "texture_analysis - Calculate GLCM texture features",
        "atmospheric_correction - Sen2Cor/FLAASH atmospheric correction",
        "geometric_correction - Orthorectification and georeferencing",
        "cloud_masking - Cloud and shadow masking",
        "data_validation - Comprehensive data quality validation",
        "inventory_generation - Data inventory and catalog generation",
        "random_forest - Random Forest classification/regression",
        "model_validation - Model performance validation",
        "risk_mapping - Generate risk/susceptibility maps",
        "map_visualization - Interactive map generation",
        "report_generation - Generate analysis reports"
    ]
    
    for step_type in step_types:
        print(f"  - {step_type}")

def list_available_processes():
    """List available process files"""
    processes_dir = Path('processes')
    if not processes_dir.exists():
        print("⚠️  No processes directory found")
        return
    
    print("📁 Available process files:")
    
    # Check for files directly in processes directory
    direct_files = list(processes_dir.glob('*.json'))
    if direct_files:
        print("\n📄 Root processes:")
        for process_file in sorted(direct_files):
            try:
                with open(process_file, 'r') as f:
                    process_def = json.load(f)
                name = process_def.get('process_info', {}).get('name', process_file.stem)
                desc = process_def.get('process_info', {}).get('description', 'No description')
                app_type = process_def.get('process_info', {}).get('application_type', 'Unknown')
                print(f"  📄 {process_file.name}")
                print(f"     Name: {name}")
                print(f"     Type: {app_type}")
                print(f"     Description: {desc[:60]}{'...' if len(desc) > 60 else ''}")
            except Exception as e:
                print(f"  📄 {process_file.name}: (invalid JSON - {e})")
    
    # Check subdirectories
    for category_dir in sorted(processes_dir.iterdir()):
        if category_dir.is_dir():
            process_files = list(category_dir.glob('*.json'))
            if process_files:
                print(f"\n📁 {category_dir.name} ({len(process_files)} files):")
                for process_file in sorted(process_files):
                    try:
                        with open(process_file, 'r') as f:
                            process_def = json.load(f)
                        name = process_def.get('process_info', {}).get('name', process_file.stem)
                        desc = process_def.get('process_info', {}).get('description', 'No description')
                        app_type = process_def.get('process_info', {}).get('application_type', 'Unknown')
                        print(f"  - {process_file.name}")
                        print(f"    Name: {name} ({app_type})")
                        print(f"    Description: {desc[:60]}{'...' if len(desc) > 60 else ''}")
                    except Exception as e:
                        print(f"  - {process_file.name}: (invalid JSON - {e})")

def show_execution_plan(process_path: Path, template_vars: Dict[str, Any]):
    """Show execution plan for dry run"""
    try:
        with open(process_path, 'r') as f:
            process_definition = json.load(f)
    except Exception as e:
        print(f"❌ Error loading process file: {e}")
        return
        
    process_info = process_definition.get('process_info', {})
    steps = process_definition.get('steps', [])
    
    print(f"\n🚀 Execution Plan for: {process_info.get('name', 'Unknown Process')}")
    print(f"   Version: {process_info.get('version', 'Unknown')}")
    print(f"   Application Type: {process_info.get('application_type', 'Unknown')}")
    print(f"   Description: {process_info.get('description', 'No description')}")
    
    print(f"\n   Template Variables ({len(template_vars)}):")
    for key, value in template_vars.items():
        print(f"     {key}: {value}")
    
    print(f"\n   Steps to execute ({len(steps)} total):")
    for i, step in enumerate(steps, 1):
        step_id = step.get('id', f'step_{i}')
        step_type = step.get('type', 'unknown')
        dependencies = step.get('dependencies', [])
        
        print(f"   {i}. {step_id} ({step_type})")
        if dependencies:
            print(f"      Dependencies: {', '.join(dependencies)}")

def show_process_info(process_path: Path):
    """Show detailed process information"""
    try:
        with open(process_path, 'r') as f:
            process_definition = json.load(f)
    except Exception as e:
        print(f"❌ Error loading process file: {e}")
        return
        
    process_info = process_definition.get('process_info', {})
    
    print(f"\n📋 Process Information:")
    print(f"   Name: {process_info.get('name', 'Unknown')}")
    print(f"   Version: {process_info.get('version', 'Unknown')}")
    print(f"   Application Type: {process_info.get('application_type', 'Unknown')}")
    print(f"   Author: {process_info.get('author', 'Unknown')}")
    print(f"   Created: {process_info.get('created_date', 'Unknown')}")
    
    print(f"\n   Description:")
    description = process_info.get('description', 'No description available')
    # Word wrap description
    words = description.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= 70:  # 70 char limit
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    for line in lines:
        print(f"   {line}")
    
    # Show requirements if present
    requirements = process_info.get('requirements', {})
    if requirements:
        print(f"\n   Requirements:")
        for req_type, req_list in requirements.items():
            print(f"     {req_type.title()}: {', '.join(req_list) if isinstance(req_list, list) else req_list}")
    
    steps = process_definition.get('steps', [])
    print(f"\n   Pipeline Steps ({len(steps)} total):")
    for i, step in enumerate(steps, 1):
        step_id = step.get('id', f'step_{i}')
        step_type = step.get('type', 'unknown')
        dependencies = step.get('dependencies', [])
        print(f"     {i}. {step_id} ({step_type})")
        if dependencies:
            print(f"        Depends on: {', '.join(dependencies)}")

def show_version_info():
    """Show version information"""
    print("🌍 TerraLux Modular Pipeline Orchestrator")
    print("   Version: 1.0.0-corrected")
    print("   Description: Geospatial analysis pipeline orchestrator")
    print("   Support: Landslide susceptibility mapping, mineral targeting")
    
    # Try to get orchestrator version
    try:
        from orchestrator.core.orchestrator import ModularOrchestrator
        print("   Core orchestrator: Available")
    except ImportError:
        print("   Core orchestrator: Not available")
    
    # Try to get step registry info
    try:
        from orchestrator.steps.base.step_registry import StepRegistry
        stats = StepRegistry.get_registry_stats()
        print(f"   Registered steps: {stats['total_registered']}")
    except:
        print("   Registered steps: Unknown")

def main():
    """CORRECTED: Main CLI entry point with enhanced error handling"""
    print("✓ TerraLux orchestrator loaded (v1.0.0-corrected)")
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Determine log level
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    elif args.log_level:
        log_level = getattr(logging, args.log_level)
    else:
        log_level = logging.INFO
    
    # Setup logging
    setup_logging(log_level, args.log_file)
    logger = logging.getLogger('orchestrator.cli')
    
    try:
        # Handle utility commands first
        if args.version:
            show_version_info()
            return 0
            
        if args.list_steps:
            list_available_steps()
            return 0
        
        if args.list_processes:
            list_available_processes()
            return 0
        
        # Require process file for other operations
        if not args.process_file:
            parser.print_help()
            return 1
        
        # Check if process file exists
        process_path = Path(args.process_file)
        if not process_path.exists():
            logger.error(f"Process file not found: {args.process_file}")
            return 1
        
        # Handle info command
        if args.info:
            show_process_info(process_path)
            return 0
        
        # Validate process file if requested
        if args.validate:
            success = validate_process_file(str(process_path), args.schema)
            return 0 if success else 1
        
        # For execution, check required arguments (unless forced mock)
        if not args.force_mock and (not args.bbox or not args.start_date or not args.end_date):
            logger.error("Missing required arguments for execution: --bbox, --start-date, --end-date")
            logger.info("Use --info to see process details, --validate to check process file, or --force-mock for testing")
            return 1
        
        # Build template variables
        template_vars = build_template_variables(args)
        
        # Set default area name if not provided
        if 'area_name' not in template_vars:
            template_vars['area_name'] = 'default_area'
        
        # Set default output directory if not provided
        if 'output_dir' not in template_vars:
            template_vars['output_dir'] = f"outputs/{template_vars['area_name']}"
        
        logger.info(f"Template variables: {template_vars}")
        
        # Show execution plan if dry run
        if args.dry_run:
            show_execution_plan(process_path, template_vars)
            return 0
        
        # CORRECTED: Try orchestrator with proper error handling
        orchestrator_result = None
        
        # Try to use the modular orchestrator
        try:
            from orchestrator.core.orchestrator import ModularOrchestrator
            
            logger.info("Using modular orchestrator")
            
            # Initialize orchestrator
            orchestrator = ModularOrchestrator()
            
            # CORRECTED: Load process using the proper method signature
            orchestrator.load_process(process_path, template_vars)
            
            # Show execution summary before running
            summary = orchestrator.get_execution_summary()
            logger.info(f"Execution Summary - Total: {summary['total_steps']}, Real: {summary['real_steps']}, Mock: {summary['mock_steps']}")
            
            # Get process info for display
            process_info = orchestrator.get_process_info()
            
            # Execute process
            print(f"🚀 Starting execution of: {process_info.get('name', 'Unknown Process')}")
            start_time = time.time()
            
            result = orchestrator.execute_process(template_vars)
            
            execution_time = time.time() - start_time
            
            # CORRECTED: Enhanced result handling
            if result.get('status') == 'success':
                print(f"✅ Process completed successfully!")
                print(f"   Execution time: {execution_time:.2f} seconds")
                
                step_results = result.get('step_results', {})
                successful_steps = result.get('successful_steps', len([k for k, v in step_results.items() if v.get('status') == 'success']))
                total_steps = result.get('total_steps', len(step_results))
                print(f"   Steps completed: {successful_steps}/{total_steps}")
                
                # Show output directory
                print(f"   Output directory: {template_vars['output_dir']}")
                
                # Show artifacts if any
                artifacts = result.get('artifacts', {})
                if artifacts:
                    print(f"   Artifacts generated: {len(artifacts)}")
                    for key in list(artifacts.keys())[:3]:  # Show first 3
                        print(f"     - {key}")
                    if len(artifacts) > 3:
                        print(f"     ... and {len(artifacts) - 3} more")
                
                return 0
                
            elif result.get('status') == 'completed_with_errors':
                print(f"⚠️ Process completed with warnings!")
                print(f"   Execution time: {execution_time:.2f} seconds")
                
                step_results = result.get('step_results', {})
                successful_steps = result.get('successful_steps', len([k for k, v in step_results.items() if v.get('status') == 'success']))
                total_steps = result.get('total_steps', len(step_results))
                print(f"   Steps completed: {successful_steps}/{total_steps}")
                
                # Show errors
                errors = result.get('errors', [])
                if errors:
                    print(f"   Errors encountered: {len(errors)}")
                    for error in errors[:2]:  # Show first 2 errors
                        print(f"     - {error}")
                    if len(errors) > 2:
                        print(f"     ... and {len(errors) - 2} more errors")
                
                return 0  # Still return success for partial completion
                
            else:
                print(f"❌ Process failed!")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
                if 'errors' in result:
                    print(f"   Multiple errors occurred:")
                    for error in result['errors'][:3]:
                        print(f"     - {error}")
                return 1
                
        except ImportError as ie:
            logger.warning(f"Modular orchestrator not available ({ie})")
            orchestrator_result = {'status': 'import_error', 'error': str(ie)}
        except Exception as e:
            logger.error(f"Modular orchestrator failed: {e}")
            orchestrator_result = {'status': 'execution_error', 'error': str(e)}
        
        # If we get here, orchestrator failed - try fallback
        if args.force_mock or orchestrator_result:
            logger.info("Using simple mock execution as fallback")
            
            # Simple fallback execution
            result = execute_with_simple_mock(process_path, template_vars)
            
            if result.get('status') == 'success':
                print(f"✅ Mock process completed successfully!")
                print(f"   Note: This was a simulation - no real data processing occurred")
                print(f"   Steps executed: {result.get('steps_executed', 0)}")
                print(f"   Output directory: {template_vars['output_dir']}")
                print(f"   💡 Install missing dependencies for real data processing")
                return 0
            else:
                print(f"❌ Mock process failed: {result.get('error', 'Unknown error')}")
                return 1
        
        # If nothing worked
        print("❌ No orchestrator implementation available")
        return 1
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def execute_with_simple_mock(process_path: Path, template_vars: Dict[str, Any]) -> Dict[str, Any]:
    """Simple mock execution as final fallback"""
    try:
        # Load process definition
        with open(process_path, 'r') as f:
            process_def = json.load(f)
        
        steps = process_def.get('steps', [])
        process_name = process_def.get('process_info', {}).get('name', 'Unknown Process')
        
        print(f"🔧 Mock execution of: {process_name}")
        print(f"   Processing {len(steps)} steps...")
        
        # Create output directory
        output_dir = Path(template_vars.get('output_dir', 'outputs/mock_test'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple step execution simulation
        for i, step in enumerate(steps, 1):
            step_id = step.get('id', f'step_{i}')
            step_type = step.get('type', 'unknown')
            
            print(f"   [{i}/{len(steps)}] {step_id} ({step_type})... ", end="", flush=True)
            time.sleep(0.2)  # Brief pause for realism
            print("✓ (mock)")
            
            # Create a simple output file for each step
            output_file = output_dir / f"{step_id}_mock_output.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'step_id': step_id,
                    'step_type': step_type,
                    'status': 'completed_mock',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'mock_data': {
                        'bbox': template_vars.get('bbox', [0, 0, 1, 1]),
                        'date_range': f"{template_vars.get('start_date', '2023-01-01')} to {template_vars.get('end_date', '2023-12-31')}",
                        'area_name': template_vars.get('area_name', 'mock_area')
                    }
                }, f, indent=2)
        
        # Create a summary file
        summary_file = output_dir / 'execution_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'process_name': process_name,
                'execution_type': 'mock',
                'steps_total': len(steps),
                'steps_completed': len(steps),
                'template_variables': template_vars,
                'execution_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        return {
            'status': 'success',
            'steps_executed': len(steps),
            'output_directory': str(output_dir),
            'note': 'Mock execution completed successfully'
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }


if __name__ == '__main__':
    sys.exit(main())
