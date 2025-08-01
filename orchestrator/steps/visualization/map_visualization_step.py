"""
Interactive map visualization step using Folium for web-based mapping.

This step creates interactive web maps for landslide susceptibility and mineral
targeting results with multiple layers, custom styling, and rich popup information.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from branca.colormap import ColorMap
from branca.element import Template, MacroElement
import json
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from ..base.base_step import BaseStep

logger = logging.getLogger(__name__)


class MapVisualizationStep(BaseStep):
    """
    Interactive map visualization step using Folium.
    
    Creates comprehensive web-based interactive maps for landslide susceptibility
    and mineral targeting with multiple data layers, custom styling, and analysis tools.
    """
    
    def __init__(self):
        super().__init__()
        self.map_object_: Optional[folium.Map] = None
        self.layer_info_: Dict[str, Any] = {}
        
    def get_step_type(self) -> str:
        return "map_visualization"
    
    def get_required_inputs(self) -> list:
        return ['prediction_results']
    
    def get_outputs(self) -> list:
        return ['interactive_map', 'map_metadata']
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Validate map visualization hyperparameters."""
        # Validate application type
        application_type = hyperparameters.get('application_type', 'landslide_susceptibility')
        valid_types = ['landslide_susceptibility', 'mineral_targeting', 'generic']
        if application_type not in valid_types:
            logger.error(f"Invalid application_type: {application_type}. Must be one of {valid_types}")
            return False
        
        # Validate map center
        map_center = hyperparameters.get('map_center', None)
        if map_center and (not isinstance(map_center, list) or len(map_center) != 2):
            logger.error("map_center must be a list of [latitude, longitude]")
            return False
        
        # Validate zoom level
        zoom_start = hyperparameters.get('zoom_start', 12)
        if not isinstance(zoom_start, int) or not (1 <= zoom_start <= 18):
            logger.error("zoom_start must be an integer between 1 and 18")
            return False
        
        # Validate colormap
        colormap = hyperparameters.get('colormap', 'viridis')
        valid_colormaps = ['viridis', 'plasma', 'RdYlBu_r', 'Spectral_r', 'YlOrRd', 'Reds', 'Blues']
        if colormap not in valid_colormaps:
            logger.warning(f"Colormap '{colormap}' may not be supported. Recommended: {valid_colormaps}")
        
        return True
    
    def execute(self, context) -> Dict[str, Any]:
        """
        Execute interactive map visualization.
        
        Args:
            context: Pipeline context containing inputs and configuration
            
        Returns:
            Dictionary containing map outputs and metadata
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
            
            # Load prediction results
            prediction_data = self._load_prediction_data(
                context['inputs']['prediction_results'], hyperparameters
            )
            
            if not prediction_data:
                return {
                    'status': 'failed',
                    'error': 'Failed to load prediction data',
                    'outputs': {}
                }
            
            # Create base map
            base_map = self._create_base_map(prediction_data, hyperparameters)
            
            # Add data layers
            self._add_prediction_layers(base_map, prediction_data, hyperparameters)
            
            # Add optional layers
            optional_layers = self._add_optional_layers(
                base_map, context.get('inputs', {}), hyperparameters
            )
            
            # Add interactive controls
            self._add_map_controls(base_map, hyperparameters)
            
            # Add custom styling and legends
            self._add_custom_styling(base_map, prediction_data, hyperparameters)
            
            # Save map
            output_path = self._save_map(base_map, context, hyperparameters)
            
            # Generate map metadata
            map_metadata = self._generate_map_metadata(
                base_map, prediction_data, optional_layers, hyperparameters
            )
            
            # Store results
            self.map_object_ = base_map
            self.layer_info_ = map_metadata.get('layers', {})
            
            # Prepare outputs
            outputs = {
                'interactive_map': {
                    'file_path': output_path,
                    'map_object': base_map,
                    'format': 'html'
                },
                'map_metadata': map_metadata
            }
            
            logger.info("Interactive map visualization completed successfully")
            logger.info(f"Map saved to: {output_path}")
            
            return {
                'status': 'success',
                'message': 'Interactive map created successfully',
                'outputs': outputs,
                'metadata': {
                    'execution_time': context.get('execution_time', 0),
                    'map_bounds': map_metadata.get('bounds'),
                    'n_layers': len(map_metadata.get('layers', {})),
                    'application_type': hyperparameters.get('application_type', 'generic')
                }
            }
            
        except Exception as e:
            logger.error(f"Map visualization failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'outputs': {}
            }
    
    def _load_prediction_data(self, prediction_input: Union[str, Dict[str, Any]], 
                            hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Load prediction data from various sources."""
        prediction_data = {}
        
        try:
            if isinstance(prediction_input, dict):
                # Data passed directly
                prediction_data = prediction_input
            elif isinstance(prediction_input, str):
                # Load from file path
                if prediction_input.endswith('.geojson'):
                    gdf = gpd.read_file(prediction_input)
                    prediction_data['vector_data'] = gdf
                elif prediction_input.endswith(('.tif', '.tiff')):
                    with rasterio.open(prediction_input) as src:
                        raster_data = src.read(1)
                        prediction_data['raster_data'] = raster_data
                        prediction_data['raster_transform'] = src.transform
                        prediction_data['raster_crs'] = src.crs
                        prediction_data['raster_bounds'] = src.bounds
                else:
                    logger.error(f"Unsupported prediction data format: {prediction_input}")
                    return {}
            
            # Extract bounds for map centering
            if 'vector_data' in prediction_data:
                bounds = prediction_data['vector_data'].total_bounds
                prediction_data['bounds'] = bounds
            elif 'raster_bounds' in prediction_data:
                prediction_data['bounds'] = prediction_data['raster_bounds']
            
            logger.info("Loaded prediction data successfully")
            return prediction_data
            
        except Exception as e:
            logger.error(f"Failed to load prediction data: {str(e)}")
            return {}
    
    def _create_base_map(self, prediction_data: Dict[str, Any], 
                        hyperparameters: Dict[str, Any]) -> folium.Map:
        """Create base Folium map with appropriate center and zoom."""
        # Determine map center
        if hyperparameters.get('map_center'):
            map_center = hyperparameters['map_center']
        elif 'bounds' in prediction_data:
            bounds = prediction_data['bounds']
            map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        else:
            map_center = [0, 0]  # Default center
        
        # Create base map
        zoom_start = hyperparameters.get('zoom_start', 12)
        
        m = folium.Map(
            location=map_center,
            zoom_start=zoom_start,
            tiles=None  # We'll add custom tiles
        )
        
        # Add base tile layers
        self._add_base_tiles(m, hyperparameters)
        
        return m
    
    def _add_base_tiles(self, map_obj: folium.Map, hyperparameters: Dict[str, Any]):
        """Add base tile layers to the map."""
        base_tiles = hyperparameters.get('base_tiles', ['OpenStreetMap', 'Satellite', 'Terrain'])
        
        # Default OpenStreetMap
        if 'OpenStreetMap' in base_tiles:
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='Street Map',
                control=True
            ).add_to(map_obj)
        
        # Satellite imagery
        if 'Satellite' in base_tiles:
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri WorldImagery',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(map_obj)
        
        # Terrain
        if 'Terrain' in base_tiles:
            folium.TileLayer(
                tiles='Stamen Terrain',
                name='Terrain',
                overlay=False,
                control=True
            ).add_to(map_obj)
        
        # Topographic
        if 'Topographic' in base_tiles:
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
                attr='Esri World Topographic Map',
                name='Topographic',
                overlay=False,
                control=True
            ).add_to(map_obj)
    
    def _add_prediction_layers(self, map_obj: folium.Map, prediction_data: Dict[str, Any],
                             hyperparameters: Dict[str, Any]):
        """Add prediction result layers to the map."""
        application_type = hyperparameters.get('application_type', 'generic')
        
        # Add vector prediction data
        if 'vector_data' in prediction_data:
            self._add_vector_layer(map_obj, prediction_data['vector_data'], application_type, hyperparameters)
        
        # Add raster prediction data
        if 'raster_data' in prediction_data:
            self._add_raster_layer(
                map_obj, 
                prediction_data['raster_data'],
                prediction_data.get('raster_transform'),
                prediction_data.get('raster_bounds'),
                application_type,
                hyperparameters
            )
    
    def _add_vector_layer(self, map_obj: folium.Map, gdf: gpd.GeoDataFrame, 
                         application_type: str, hyperparameters: Dict[str, Any]):
        """Add vector prediction layer with appropriate styling."""
        # Determine the prediction column
        prediction_columns = ['prediction', 'class', 'susceptibility', 'prospectivity', 'risk_level']
        prediction_col = None
        
        for col in prediction_columns:
            if col in gdf.columns:
                prediction_col = col
                break
        
        if prediction_col is None:
            logger.warning("No prediction column found in vector data")
            return
        
        # Get styling parameters
        colormap_name = hyperparameters.get('colormap', 'viridis')
        
        # Create color mapping
        if application_type == 'landslide_susceptibility':
            color_map = self._create_landslide_colormap(gdf[prediction_col], colormap_name)
            layer_name = 'Landslide Susceptibility'
        elif application_type == 'mineral_targeting':
            color_map = self._create_mineral_colormap(gdf[prediction_col], colormap_name)
            layer_name = 'Mineral Prospectivity'
        else:
            color_map = self._create_generic_colormap(gdf[prediction_col], colormap_name)
            layer_name = 'Prediction Results'
        
        # Add to map
        folium.GeoJson(
            gdf,
            style_function=lambda feature: {
                'fillColor': color_map.get(feature['properties'].get(prediction_col, 0), '#gray'),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7,
                'opacity': 0.8
            },
            popup=folium.GeoJsonPopup(
                fields=list(gdf.columns),
                aliases=[col.replace('_', ' ').title() for col in gdf.columns],
                localize=True,
                labels=True
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=[prediction_col],
                aliases=[prediction_col.replace('_', ' ').title()],
                localize=True
            )
        ).add_to(map_obj)
        
        # Add legend
        self._add_vector_legend(map_obj, color_map, layer_name, application_type)
    
    def _add_raster_layer(self, map_obj: folium.Map, raster_data: np.ndarray,
                         transform: Any, bounds: Tuple[float, float, float, float],
                         application_type: str, hyperparameters: Dict[str, Any]):
        """Add raster prediction layer with appropriate styling."""
        try:
            # Convert raster to image overlay
            colormap_name = hyperparameters.get('colormap', 'viridis')
            
            # Get colormap
            if application_type == 'landslide_susceptibility':
                cmap = plt.cm.get_cmap('YlOrRd')
                layer_name = 'Landslide Susceptibility'
            elif application_type == 'mineral_targeting':
                cmap = plt.cm.get_cmap('plasma')
                layer_name = 'Mineral Prospectivity'
            else:
                cmap = plt.cm.get_cmap(colormap_name)
                layer_name = 'Prediction Results'
            
            # Normalize raster data
            raster_norm = (raster_data - np.nanmin(raster_data)) / (np.nanmax(raster_data) - np.nanmin(raster_data))
            raster_norm = np.nan_to_num(raster_norm, nan=0)
            
            # Apply colormap
            raster_colored = cmap(raster_norm)
            
            # Convert to image format
            from PIL import Image
            import io
            import base64
            
            # Convert to PIL Image
            raster_image = Image.fromarray((raster_colored * 255).astype(np.uint8))
            
            # Save to bytes for embedding
            img_buffer = io.BytesIO()
            raster_image.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Add as image overlay
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_str}",
                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                opacity=0.7,
                interactive=True,
                cross_origin=False,
                name=layer_name
            ).add_to(map_obj)
            
            # Add colorbar legend
            self._add_raster_legend(map_obj, raster_data, cmap, layer_name, application_type)
            
        except Exception as e:
            logger.error(f"Failed to add raster layer: {str(e)}")
    
    def _create_landslide_colormap(self, values: pd.Series, colormap_name: str) -> Dict[Any, str]:
        """Create color mapping for landslide susceptibility classes."""
        unique_values = sorted(values.unique())
        
        # Standard landslide susceptibility colors
        landslide_colors = {
            'Very Low': '#2E8B57',    # Sea Green
            'Low': '#FFD700',         # Gold
            'Moderate': '#FF8C00',    # Dark Orange
            'High': '#FF4500',        # Orange Red
            'Very High': '#DC143C',   # Crimson
            0: '#2E8B57',
            1: '#FFD700',
            2: '#FF8C00',
            3: '#FF4500',
            4: '#DC143C'
        }
        
        color_map = {}
        for i, value in enumerate(unique_values):
            if value in landslide_colors:
                color_map[value] = landslide_colors[value]
            elif i < len(landslide_colors):
                color_map[value] = landslide_colors[i]
            else:
                # Fallback to matplotlib colormap
                cmap = plt.cm.get_cmap('YlOrRd')
                color_map[value] = mcolors.to_hex(cmap(i / len(unique_values)))
        
        return color_map
    
    def _create_mineral_colormap(self, values: pd.Series, colormap_name: str) -> Dict[Any, str]:
        """Create color mapping for mineral prospectivity classes."""
        unique_values = sorted(values.unique())
        
        # Standard mineral prospectivity colors
        mineral_colors = {
            'Background': '#2F4F4F',      # Dark Slate Gray
            'Low Potential': '#4682B4',    # Steel Blue
            'Moderate Potential': '#DAA520', # Goldenrod
            'High Potential': '#DC143C',   # Crimson
            0: '#2F4F4F',
            1: '#4682B4',
            2: '#DAA520',
            3: '#DC143C'
        }
        
        color_map = {}
        for i, value in enumerate(unique_values):
            if value in mineral_colors:
                color_map[value] = mineral_colors[value]
            elif i < len(mineral_colors):
                color_map[value] = mineral_colors[i]
            else:
                # Fallback to matplotlib colormap
                cmap = plt.cm.get_cmap('plasma')
                color_map[value] = mcolors.to_hex(cmap(i / len(unique_values)))
        
        return color_map
    
    def _create_generic_colormap(self, values: pd.Series, colormap_name: str) -> Dict[Any, str]:
        """Create generic color mapping."""
        unique_values = sorted(values.unique())
        cmap = plt.cm.get_cmap(colormap_name)
        
        color_map = {}
        for i, value in enumerate(unique_values):
            color_map[value] = mcolors.to_hex(cmap(i / len(unique_values)))
        
        return color_map
    
    def _add_optional_layers(self, map_obj: folium.Map, additional_inputs: Dict[str, Any],
                           hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add optional layers like training data, validation points, etc."""
        optional_layers = {}
        
        # Add training data points
        if 'training_data' in additional_inputs:
            try:
                training_layer = self._add_training_data_layer(
                    map_obj, additional_inputs['training_data'], hyperparameters
                )
                optional_layers['training_data'] = training_layer
            except Exception as e:
                logger.warning(f"Failed to add training data layer: {str(e)}")
        
        # Add validation data points
        if 'validation_data' in additional_inputs:
            try:
                validation_layer = self._add_validation_data_layer(
                    map_obj, additional_inputs['validation_data'], hyperparameters
                )
                optional_layers['validation_data'] = validation_layer
            except Exception as e:
                logger.warning(f"Failed to add validation data layer: {str(e)}")
        
        # Add uncertainty layer
        if 'uncertainty_map' in additional_inputs:
            try:
                uncertainty_layer = self._add_uncertainty_layer(
                    map_obj, additional_inputs['uncertainty_map'], hyperparameters
                )
                optional_layers['uncertainty'] = uncertainty_layer
            except Exception as e:
                logger.warning(f"Failed to add uncertainty layer: {str(e)}")
        
        # Add inventory data (landslides, mineral occurrences)
        if 'inventory_data' in additional_inputs:
            try:
                inventory_layer = self._add_inventory_layer(
                    map_obj, additional_inputs['inventory_data'], hyperparameters
                )
                optional_layers['inventory'] = inventory_layer
            except Exception as e:
                logger.warning(f"Failed to add inventory layer: {str(e)}")
        
        return optional_layers
    
    def _add_training_data_layer(self, map_obj: folium.Map, training_data: Any,
                               hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add training data points to the map."""
        # This is a simplified implementation
        # In practice, you'd load and process the training data appropriately
        return {'status': 'placeholder', 'description': 'Training data layer'}
    
    def _add_validation_data_layer(self, map_obj: folium.Map, validation_data: Any,
                                 hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add validation data points to the map."""
        return {'status': 'placeholder', 'description': 'Validation data layer'}
    
    def _add_uncertainty_layer(self, map_obj: folium.Map, uncertainty_data: Any,
                             hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add uncertainty/confidence layer to the map."""
        return {'status': 'placeholder', 'description': 'Uncertainty layer'}
    
    def _add_inventory_layer(self, map_obj: folium.Map, inventory_data: Any,
                           hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add known inventory points (landslides/mineral occurrences) to the map."""
        return {'status': 'placeholder', 'description': 'Inventory layer'}
    
    def _add_map_controls(self, map_obj: folium.Map, hyperparameters: Dict[str, Any]):
        """Add interactive controls to the map."""
        # Layer control
        folium.LayerControl(collapsed=False).add_to(map_obj)
        
        # Full screen control
        plugins.Fullscreen(
            position='topright',
            title='Full Screen',
            title_cancel='Exit Full Screen',
            force_separate_button=True
        ).add_to(map_obj)
        
        # Measure tool
        plugins.MeasureControl(
            position='topleft',
            primary_length_unit='kilometers',
            primary_area_unit='sqkilometers'
        ).add_to(map_obj)
        
        # Mouse position
        plugins.MousePosition(
            position='bottomright',
            separator=' | ',
            prefix='Coordinates:'
        ).add_to(map_obj)
        
        # Minimap
        if hyperparameters.get('add_minimap', False):
            minimap = plugins.MiniMap(toggle_display=True)
            map_obj.add_child(minimap)
    
    def _add_vector_legend(self, map_obj: folium.Map, color_map: Dict[Any, str],
                          layer_name: str, application_type: str):
        """Add legend for vector data."""
        # Create HTML legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>{layer_name}</h4>
        '''
        
        for value, color in color_map.items():
            legend_html += f'<p><i class="fa fa-square" style="color:{color}"></i> {value}</p>'
        
        legend_html += '</div>'
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_raster_legend(self, map_obj: folium.Map, raster_data: np.ndarray,
                          cmap: Any, layer_name: str, application_type: str):
        """Add colorbar legend for raster data."""
        try:
            # Create colorbar
            colormap = ColorMap(
                colors=[mcolors.to_hex(cmap(i/10)) for i in range(11)],
                vmin=float(np.nanmin(raster_data)),
                vmax=float(np.nanmax(raster_data)),
                caption=layer_name
            )
            
            map_obj.add_child(colormap)
            
        except Exception as e:
            logger.warning(f"Failed to add raster legend: {str(e)}")
    
    def _add_custom_styling(self, map_obj: folium.Map, prediction_data: Dict[str, Any],
                          hyperparameters: Dict[str, Any]):
        """Add custom CSS styling and branding."""
        # Custom CSS for better appearance
        custom_css = """
        <style>
        .leaflet-control-layers {
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 1px 7px rgba(0,0,0,0.4);
        }
        .leaflet-popup-content {
            font-family: Arial, sans-serif;
            line-height: 1.4;
        }
        .legend {
            line-height: 18px;
            color: #555;
        }
        .legend i {
            width: 18px;
            height: 18px;
            float: left;
            margin-right: 8px;
            opacity: 0.7;
        }
        </style>
        """
        
        map_obj.get_root().html.add_child(folium.Element(custom_css))
        
        # Add title/header if requested
        if hyperparameters.get('add_title', True):
            application_type = hyperparameters.get('application_type', 'generic')
            
            if application_type == 'landslide_susceptibility':
                title = 'Landslide Susceptibility Assessment'
            elif application_type == 'mineral_targeting':
                title = 'Mineral Prospectivity Mapping'
            else:
                title = 'Prediction Results'
            
            title_html = f'''
            <div style="position: fixed; 
                        top: 10px; left: 50%; transform: translateX(-50%); 
                        background-color: rgba(255, 255, 255, 0.8); 
                        border: 2px solid grey; z-index: 9999; 
                        font-size: 18px; font-weight: bold; 
                        padding: 10px; border-radius: 5px;">
                {title}
            </div>
            '''
            
            map_obj.get_root().html.add_child(folium.Element(title_html))
    
    def _save_map(self, map_obj: folium.Map, context: Dict[str, Any],
                  hyperparameters: Dict[str, Any]) -> str:
        """Save the interactive map to file."""
        output_dir = context.get('output_dir', 'outputs/visualization')
        os.makedirs(output_dir, exist_ok=True)
        
        map_filename = hyperparameters.get('map_filename', 'interactive_map.html')
        output_path = os.path.join(output_dir, map_filename)
        
        # Save map
        map_obj.save(output_path)
        
        logger.info(f"Interactive map saved to: {output_path}")
        return output_path
    
    def _generate_map_metadata(self, map_obj: folium.Map, prediction_data: Dict[str, Any],
                             optional_layers: Dict[str, Any], 
                             hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata about the created map."""
        metadata = {
            'map_type': 'interactive_folium',
            'application_type': hyperparameters.get('application_type', 'generic'),
            'bounds': prediction_data.get('bounds', []),
            'center': hyperparameters.get('map_center', []),
            'zoom_level': hyperparameters.get('zoom_start', 12),
            'layers': {
                'base_layers': hyperparameters.get('base_tiles', ['OpenStreetMap']),
                'data_layers': [],
                'optional_layers': list(optional_layers.keys())
            },
            'styling': {
                'colormap': hyperparameters.get('colormap', 'viridis'),
                'has_legend': True,
                'has_title': hyperparameters.get('add_title', True)
            },
            'interactivity': {
                'layer_control': True,
                'fullscreen': True,
                'measure_tool': True,
                'mouse_position': True,
                'minimap': hyperparameters.get('add_minimap', False)
            }
        }
        
        # Add data layer information
        if 'vector_data' in prediction_data:
            metadata['layers']['data_layers'].append('vector_predictions')
        if 'raster_data' in prediction_data:
            metadata['layers']['data_layers'].append('raster_predictions')
        
        return metadata
    
    def get_map_object(self) -> Optional[folium.Map]:
        """Get the created Folium map object."""
        return self.map_object_
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about map layers."""
        return self.layer_info_
