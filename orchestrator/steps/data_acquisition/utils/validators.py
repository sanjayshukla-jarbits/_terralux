"""
Validation utilities for Sentinel Hub requests
"""
from typing import List
from datetime import datetime

def validate_bbox(bbox: List[float]) -> List[str]:
    """Validate bounding box coordinates"""
    errors = []
    
    if len(bbox) != 4:
        errors.append("Bounding box must have 4 coordinates [west, south, east, north]")
        return errors
    
    west, south, east, north = bbox
    
    if not (-180 <= west <= 180) or not (-180 <= east <= 180):
        errors.append("Longitude must be between -180 and 180")
    if not (-90 <= south <= 90) or not (-90 <= north <= 90):
        errors.append("Latitude must be between -90 and 90")
    if west >= east:
        errors.append("West must be less than east")
    if south >= north:
        errors.append("South must be less than north")
    
    return errors

def validate_date_range(start_date: str, end_date: str) -> List[str]:
    """Validate date range"""
    errors = []
    
    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start >= end:
            errors.append("Start date must be before end date")
        
        if end > datetime.now():
            errors.append("End date cannot be in the future")
            
    except ValueError as e:
        errors.append(f"Invalid date format: {e}")
    
    return errors

def validate_bands(bands: List[str], collection: str) -> List[str]:
    """Validate band selection for data collection"""
    errors = []
    
    valid_bands = {
        'SENTINEL-2-L2A': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
        'SENTINEL-2-L1C': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
        'SENTINEL-1-GRD': ['VV', 'VH', 'HH', 'HV']
    }
    
    if collection not in valid_bands:
        errors.append(f"Unknown collection: {collection}")
        return errors
    
    invalid_bands = [band for band in bands if band not in valid_bands[collection]]
    if invalid_bands:
        errors.append(f"Invalid bands for {collection}: {invalid_bands}")
    
    return errors
