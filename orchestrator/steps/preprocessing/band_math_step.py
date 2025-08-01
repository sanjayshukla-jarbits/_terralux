# orchestrator/steps/preprocessing/band_math_step.py
"""
Generic band mathematics step for spectral calculations and transformations.

Supports custom mathematical operations, spectral indices, band combinations,
and arithmetic operations across multiple bands and images.
"""

import rasterio
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import logging
from datetime import datetime
import re
import ast
import operator
from functools import reduce

from ..base.base_step import BaseStep
from ..base.step_registry import StepRegistry
from ...utils.file_utils import ensure_directory, validate_file_exists


class BandMathStep(BaseStep):
    """
    Universal band mathematics step for spectral calculations.
    
    Capabilities:
    - Custom mathematical expressions
    - Predefined spectral indices
    - Multi-image operations
    - Statistical operations
    - Conditional operations
    - Safe expression evaluation
    
    Configuration Examples:
    
    For Landslide Assessment:
    {
        "operations": [
            {
                "name": "NDVI",
                "expression": "(B8 - B4) / (B8 + B4)",
                "description": "Normalized Difference Vegetation Index"
            },
            {
                "name": "NDWI", 
                "expression": "(B3 - B8) / (B3 + B8)",
                "description": "Normalized Difference Water Index"
            }
        ],
        "output_format": "multiband",
        "data_type": "float32"
    }
    
    For Mineral Targeting:
    {
        "operations": [
            {
                "name": "clay_ratio",
                "expression": "B11 / B12", 
                "description": "Clay mineral ratio"
            },
            {
                "name": "iron_oxide",
                "expression": "B4 / B2",
                "description": "Iron oxide ratio"
            },
            {
                "name": "alteration_index",
                "expression": "(B11 + B12) / (B8 + B4)",
                "description": "Alteration mapping index"
            }
        ],
        "mask_expression": "B8 > 0",
        "output_format": "separate_files"
    }
    """
    
    def __init__(self, step_id: str, hyperparameters: Dict[str, Any]):
        super().__init__(step_id, "band_math", hyperparameters)
        
        # Operations configuration
        self.operations = hyperparameters.get("operations", [])
        self.mask_expression = hyperparameters.get("mask_expression", None)
        self.global_mask = hyperparameters.get("global_mask", None)
        
        # Processing options
        self.data_type = hyperparameters.get("data_type", "float32")
        self.output_format = hyperparameters.get("output_format", "multiband")  # multiband, separate_files
        self.nodata_value = hyperparameters.get("nodata_value", -9999)
        self.scale_factor = hyperparameters.get("scale_factor", 1.0)
        self.offset = hyperparameters.get("offset", 0.0)
        
        # Quality control
        self.validate_expressions = hyperparameters.get("validate_expressions", True)
        self.handle_division_by_zero = hyperparameters.get("handle_division_by_zero", True)
        self.clip_output = hyperparameters.get("clip_output", None)  # [min, max]
        
        # Performance options
        self.chunk_size = hyperparameters.get("chunk_size", 1024)
        self.use_dask = hyperparameters.get("use_dask", False)
        
        # Predefined indices
        self.predefined_indices = self._get_predefined_indices()
        
        # Safe evaluation setup
        self.safe_operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        self.safe_functions = {
            'sqrt': np.sqrt,
            'abs': np.abs,
            'log': np.log,
            'log10': np.log10,
            'exp': np.exp,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
            'min': np.minimum,
            'max': np.maximum,
            'where': np.where,
            'clip': np.clip
        }
        
        self.logger = logging.getLogger(f"BandMath.{step_id}")
    
    def execute(self, context) -> Dict[str, Any]:
        """Execute band mathematics operations"""
        try:
            self.logger.info(f"Starting band mathematics with {len(self.operations)} operations")
            
            # Get input data
            input_files = self._get_input_files(context)
            
            # Validate inputs
            self._validate_inputs(input_files)
            
            # Load and prepare data
            band_data, band_metadata = self._load_band_data(input_files)
            
            # Validate expressions
            if self.validate_expressions:
                self._validate_expressions(band_data.keys())
            
            # Execute operations
            outputs = self._execute_operations(band_data, band_metadata, context)
            
            # Apply global mask if specified
            if self.global_mask:
                outputs = self._apply_global_mask(outputs, band_data, context)
            
            # Update context
            context.add_data(f"{self.step_id}_calculated_bands", outputs["output_files"])
            if "operation_metadata" in outputs:
                context.add_data(f"{self.step_id}_operation_metadata", outputs["operation_metadata"])
            
            self.logger.info("Band mathematics completed successfully")
            return {
                "status": "success",
                "outputs": outputs,
                "metadata": {
                    "operations_count": len(self.operations),
                    "output_format": self.output_format,
                    "data_type": self.data_type,
                    "processing_time": outputs.get("processing_time")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Band mathematics failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_input_files(self, context) -> List[str]:
        """Get input files from context"""
        # Try different possible data keys
        for key in ["resampled_images", "masked_images", "corrected_images", "spectral_data"]:
            data = getattr(context, 'get_data', lambda x: None)(key)
            if data:
                if isinstance(data, str):
                    return [data]
                elif isinstance(data, list):
                    return data
        
        # Check hyperparameters
        if "input_files" in self.hyperparameters:
            input_files = self.hyperparameters["input_files"]
            if isinstance(input_files, str):
                return [input_files]
            return input_files
        
        raise ValueError("No input files found for band mathematics")
    
    def _validate_inputs(self, input_files: List[str]):
        """Validate input files"""
        if not input_files:
            raise ValueError("No input files provided for band mathematics")
        
        for file_path in input_files:
            if not validate_file_exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
    
    def _get_predefined_indices(self) -> Dict[str, Dict[str, str]]:
        """Get predefined spectral indices"""
        return {
            # Vegetation indices
            "NDVI": {
                "expression": "(B8 - B4) / (B8 + B4)",
                "description": "Normalized Difference Vegetation Index"
            },
            "EVI": {
                "expression": "2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))",
                "description": "Enhanced Vegetation Index"
            },
            "SAVI": {
                "expression": "((B8 - B4) / (B8 + B4 + 0.5)) * 1.5",
                "description": "Soil Adjusted Vegetation Index"
            },
            
            # Water indices
            "NDWI": {
                "expression": "(B3 - B8) / (B3 + B8)",
                "description": "Normalized Difference Water Index"
            },
            "MNDWI": {
                "expression": "(B3 - B11) / (B3 + B11)",
                "description": "Modified NDWI"
            },
            
            # Urban/built-up indices
            "NDBI": {
                "expression": "(B11 - B8) / (B11 + B8)",
                "description": "Normalized Difference Built-up Index"
            },
            "UI": {
                "expression": "(B11 - B8) / (B11 + B8)",
                "description": "Urban Index"
            },
            
            # Soil indices
            "BSI": {
                "expression": "((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))",
                "description": "Bare Soil Index"
            },
            
            # Mineral indices
            "clay_minerals": {
                "expression": "B11 / B12",
                "description": "Clay minerals ratio"
            },
            "iron_oxides": {
                "expression": "B4 / B2",
                "description": "Iron oxides ratio"
            },
            "carbonates": {
                "expression": "B12 / B11",
                "description": "Carbonate minerals ratio"
            },
            "ferric_iron": {
                "expression": "B4 / B3",
                "description": "Ferric iron ratio"
            },
            "ferrous_iron": {
                "expression": "B12 / B11",
                "description": "Ferrous iron ratio"
            },
            "alunite": {
                "expression": "B11 / B12",
                "description": "Alunite detection ratio"
            },
            
            # Alteration indices
            "alteration_index": {
                "expression": "(B11 + B12) / (B8 + B4)",
                "description": "Hydrothermal alteration index"
            },
            "gossan_index": {
                "expression": "B4 / B8",
                "description": "Gossan detection index"
            }
        }
    
    def _load_band_data(self, input_files: List[str]) -> tuple:
        """Load band data from input files"""
        band_data = {}
        band_metadata = {}
        
        for file_idx, file_path in enumerate(input_files):
            with rasterio.open(file_path) as src:
                # Store metadata
                file_key = f"file_{file_idx}"
                band_metadata[file_key] = {
                    "file_path": file_path,
                    "profile": src.profile,
                    "transform": src.transform,
                    "crs": src.crs,
                    "bounds": src.bounds
                }
                
                # Load band data
                for band_idx in range(1, src.count + 1):
                    band_key = f"B{band_idx}"
                    if len(input_files) > 1:
                        band_key = f"F{file_idx}_B{band_idx}"
                    
                    band_array = src.read(band_idx).astype(np.float64)
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        band_array = np.where(band_array == src.nodata, np.nan, band_array)
                    
                    band_data[band_key] = band_array
                
                # If single file, also create simplified band references
                if len(input_files) == 1:
                    for band_idx in range(1, src.count + 1):
                        simplified_key = f"B{band_idx}"
                        if simplified_key not in band_data:
                            band_data[simplified_key] = band_data[f"F0_B{band_idx}"]
        
        return band_data, band_metadata
    
    def _validate_expressions(self, available_bands: List[str]):
        """Validate mathematical expressions"""
        for operation in self.operations:
            expression = operation.get("expression", "")
            
            if not expression:
                raise ValueError(f"Empty expression in operation: {operation.get('name', 'unnamed')}")
            
            # Check for band references
            band_refs = re.findall(r'B\d+', expression)
            for band_ref in band_refs:
                if band_ref not in available_bands:
                    self.logger.warning(f"Band {band_ref} referenced in expression but not available")
            
            # Validate expression syntax
            try:
                self._safe_eval(expression, {band: np.array([1.0]) for band in available_bands})
            except Exception as e:
                raise ValueError(f"Invalid expression '{expression}': {str(e)}")
    
    def _execute_operations(self, band_data: Dict[str, np.ndarray], 
                           band_metadata: Dict[str, Any], context) -> Dict[str, Any]:
        """Execute all mathematical operations"""
        processing_start = datetime.now()
        output_dir = context.get_temp_dir() / "band_math"
        ensure_directory(output_dir)
        
        results = {}
        output_files = []
        operation_metadata = []
        
        # Get reference profile for output
        ref_profile = list(band_metadata.values())[0]["profile"]
        
        # Process each operation
        for operation in self.operations:
            name = operation.get("name", f"operation_{len(results)}")
            expression = operation.get("expression", "")
            description = operation.get("description", "")
            
            self.logger.info(f"Calculating {name}: {expression}")
            
            try:
                # Evaluate expression
                result_array = self._safe_eval(expression, band_data)
                
                # Apply mask if specified
                if self.mask_expression:
                    mask = self._safe_eval(self.mask_expression, band_data)
                    result_array = np.where(mask, result_array, self.nodata_value)
                
                # Handle division by zero and invalid values
                if self.handle_division_by_zero:
                    result_array = np.where(np.isfinite(result_array), result_array, self.nodata_value)
                
                # Apply clipping if specified
                if self.clip_output:
                    min_val, max_val = self.clip_output
                    result_array = np.clip(result_array, min_val, max_val)
                
                # Apply scaling and offset
                if self.scale_factor != 1.0 or self.offset != 0.0:
                    result_array = (result_array * self.scale_factor) + self.offset
                
                results[name] = result_array
                
                # Store operation metadata
                operation_metadata.append({
                    "name": name,
                    "expression": expression,
                    "description": description,
                    "min_value": float(np.nanmin(result_array)),
                    "max_value": float(np.nanmax(result_array)),
                    "mean_value": float(np.nanmean(result_array)),
                    "valid_pixels": int(np.sum(np.isfinite(result_array)))
                })
                
            except Exception as e:
                self.logger.error(f"Failed to calculate {name}: {str(e)}")
                operation_metadata.append({
                    "name": name,
                    "expression": expression,
                    "description": description,
                    "error": str(e)
                })
        
        # Save results based on output format
        if self.output_format == "multiband":
            output_files = [self._save_multiband_result(results, ref_profile, output_dir)]
        else:  # separate_files
            for name, array in results.items():
                output_file = self._save_single_band_result(name, array, ref_profile, output_dir)
                output_files.append(output_file)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        return {
            "output_files": output_files,
            "results": results,
            "operation_metadata": operation_metadata,
            "processing_time": processing_time
        }
    
    def _safe_eval(self, expression: str, band_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Safely evaluate mathematical expression"""
        try:
            # Parse the expression
            node = ast.parse(expression, mode='eval')
            
            # Evaluate the expression
            return self._eval_node(node.body, band_data)
            
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {str(e)}")
    
    def _eval_node(self, node, band_data: Dict[str, np.ndarray]):
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Name):
            # Band reference
            if node.id in band_data:
                return band_data[node.id]
            else:
                raise ValueError(f"Unknown band reference: {node.id}")
        elif isinstance(node, ast.BinOp):
            # Binary operation
            left = self._eval_node(node.left, band_data)
            right = self._eval_node(node.right, band_data)
            op = self.safe_operations.get(type(node.op))
            if op:
                if isinstance(node.op, ast.Div) and self.handle_division_by_zero:
                    # Handle division by zero
                    return np.divide(left, right, out=np.full_like(left, np.nan), where=right!=0)
                else:
                    return op(left, right)
            else:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            # Unary operation
            operand = self._eval_node(node.operand, band_data)
            op = self.safe_operations.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
        elif isinstance(node, ast.Call):
            # Function call
            func_name = node.func.id
            if func_name in self.safe_functions:
                args = [self._eval_node(arg, band_data) for arg in node.args]
                return self.safe_functions[func_name](*args)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        elif isinstance(node, ast.Compare):
            # Comparison operation (for conditional expressions)
            left = self._eval_node(node.left, band_data)
            
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, band_data)
                
                if isinstance(op, ast.Lt):
                    result = left < right
                elif isinstance(op, ast.LtE):
                    result = left <= right
                elif isinstance(op, ast.Gt):
                    result = left > right
                elif isinstance(op, ast.GtE):
                    result = left >= right
                elif isinstance(op, ast.Eq):
                    result = left == right
                elif isinstance(op, ast.NotEq):
                    result = left != right
                else:
                    raise ValueError(f"Unsupported comparison: {type(op)}")
                
                left = result
            
            return result
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")
    
    def _save_multiband_result(self, results: Dict[str, np.ndarray], 
                              ref_profile: dict, output_dir: Path) -> str:
        """Save results as multiband image"""
        if not results:
            raise ValueError("No results to save")
        
        # Prepare output profile
        output_profile = ref_profile.copy()
        output_profile.update({
            'dtype': self.data_type,
            'count': len(results),
            'nodata': self.nodata_value
        })
        
        # Create output file
        output_file = output_dir / "band_math_results.tif"
        
        with rasterio.open(output_file, 'w', **output_profile) as dst:
            for band_idx, (name, array) in enumerate(results.items(), 1):
                # Convert to output data type
                output_array = array.astype(self.data_type)
                
                # Handle NaN values
                output_array = np.where(np.isnan(output_array), self.nodata_value, output_array)
                
                dst.write(output_array, band_idx)
                dst.set_band_description(band_idx, name)
        
        return str(output_file)
    
    def _save_single_band_result(self, name: str, array: np.ndarray, 
                                ref_profile: dict, output_dir: Path) -> str:
        """Save single band result"""
        # Prepare output profile
        output_profile = ref_profile.copy()
        output_profile.update({
            'dtype': self.data_type,
            'count': 1,
            'nodata': self.nodata_value
        })
        
        # Create output file
        safe_name = re.sub(r'[^\w\-_]', '_', name)
        output_file = output_dir / f"{safe_name}.tif"
        
        with rasterio.open(output_file, 'w', **output_profile) as dst:
            # Convert to output data type
            output_array = array.astype(self.data_type)
            
            # Handle NaN values
            output_array = np.where(np.isnan(output_array), self.nodata_value, output_array)
            
            dst.write(output_array, 1)
            dst.set_band_description(1, name)
        
        return str(output_file)
    
    def _apply_global_mask(self, outputs: Dict[str, Any], band_data: Dict[str, np.ndarray], 
                          context) -> Dict[str, Any]:
        """Apply global mask to all results"""
        try:
            # Evaluate global mask expression
            global_mask_array = self._safe_eval(self.global_mask, band_data)
            
            # Apply mask to all results
            masked_results = {}
            for name, array in outputs["results"].items():
                masked_array = np.where(global_mask_array, array, self.nodata_value)
                masked_results[name] = masked_array
            
            # Update results
            outputs["results"] = masked_results
            
            # Re-save files with mask applied
            output_dir = Path(outputs["output_files"][0]).parent
            ref_profile = list(context.get_data("band_metadata", {}).values())[0]["profile"]
            
            if self.output_format == "multiband":
                masked_file = self._save_multiband_result(masked_results, ref_profile, output_dir / "masked")
                outputs["output_files"] = [masked_file]
            else:
                masked_files = []
                for name, array in masked_results.items():
                    masked_file = self._save_single_band_result(name, array, ref_profile, output_dir / "masked")
                    masked_files.append(masked_file)
                outputs["output_files"] = masked_files
            
            # Update metadata
            for i, metadata in enumerate(outputs["operation_metadata"]):
                if "error" not in metadata:
                    name = metadata["name"]
                    if name in masked_results:
                        array = masked_results[name]
                        metadata.update({
                            "min_value": float(np.nanmin(array)),
                            "max_value": float(np.nanmax(array)),
                            "mean_value": float(np.nanmean(array)),
                            "valid_pixels": int(np.sum(np.isfinite(array))),
                            "global_mask_applied": True
                        })
            
            return outputs
            
        except Exception as e:
            self.logger.warning(f"Failed to apply global mask: {str(e)}")
            return outputs
    
    def _expand_predefined_indices(self):
        """Expand operations with predefined indices"""
        expanded_operations = []
        
        for operation in self.operations:
            name = operation.get("name", "")
            
            # Check if this is a predefined index
            if name in self.predefined_indices and "expression" not in operation:
                # Use predefined definition
                predefined = self.predefined_indices[name]
                expanded_operation = operation.copy()
                expanded_operation["expression"] = predefined["expression"]
                if "description" not in expanded_operation:
                    expanded_operation["description"] = predefined["description"]
                expanded_operations.append(expanded_operation)
            else:
                # Use as-is
                expanded_operations.append(operation)
        
        self.operations = expanded_operations
    
    def _create_operation_summary(self, operation_metadata: List[Dict[str, Any]]) -> str:
        """Create summary report of operations"""
        try:
            summary_content = f"""
Band Mathematics Operations Summary
==================================

Processing Date: {datetime.now().isoformat()}
Output Format: {self.output_format}
Data Type: {self.data_type}
NoData Value: {self.nodata_value}

Operations Executed:
"""
            
            for i, metadata in enumerate(operation_metadata, 1):
                if "error" in metadata:
                    summary_content += f"""
{i}. {metadata['name']} - FAILED
   Expression: {metadata['expression']}
   Error: {metadata['error']}
"""
                else:
                    summary_content += f"""
{i}. {metadata['name']}
   Expression: {metadata['expression']}
   Description: {metadata.get('description', 'N/A')}
   Value Range: {metadata['min_value']:.6f} to {metadata['max_value']:.6f}
   Mean Value: {metadata['mean_value']:.6f}
   Valid Pixels: {metadata['valid_pixels']:,}
"""
            
            # Calculate success rate
            successful_ops = sum(1 for m in operation_metadata if "error" not in m)
            total_ops = len(operation_metadata)
            success_rate = (successful_ops / total_ops) * 100 if total_ops > 0 else 0
            
            summary_content += f"""

Summary Statistics:
- Total Operations: {total_ops}
- Successful Operations: {successful_ops}
- Success Rate: {success_rate:.1f}%
- Global Mask Applied: {'Yes' if self.global_mask else 'No'}
- Clipping Applied: {'Yes' if self.clip_output else 'No'}
"""
            
            return summary_content
            
        except Exception as e:
            self.logger.warning(f"Failed to create operation summary: {str(e)}")
            return f"Summary creation failed: {str(e)}"


# Register the step
StepRegistry.register("band_math", BandMathStep)
