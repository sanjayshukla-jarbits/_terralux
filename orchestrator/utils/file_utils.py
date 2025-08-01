"""
File Utilities for Orchestrator
===============================

This module provides essential file system operations and utilities
for the orchestrator pipeline system.

Key Features:
- Directory creation and validation
- File existence and format validation
- Safe file operations (copy, move, delete)
- File size and hash calculations
- Temporary directory management
- File discovery and pattern matching
- Archive and compression operations
"""

import os
import shutil
import hashlib
import tempfile
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import glob
import json
from datetime import datetime
import zipfile
import tarfile

# Configure logger
logger = logging.getLogger(__name__)

# Supported file formats
SUPPORTED_RASTER_FORMATS = ['.tif', '.tiff', '.jp2', '.img', '.hdf', '.nc', '.grd']
SUPPORTED_VECTOR_FORMATS = ['.shp', '.gpkg', '.geojson', '.kml', '.gml']
SUPPORTED_CONFIG_FORMATS = ['.json', '.yaml', '.yml', '.toml']
SUPPORTED_ARCHIVE_FORMATS = ['.zip', '.tar', '.tar.gz', '.tar.bz2']


def ensure_directory(directory: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to create
        parents: Create parent directories if needed
        exist_ok: Don't raise error if directory already exists
        
    Returns:
        Path object of the created/validated directory
        
    Example:
        >>> output_dir = ensure_directory('outputs/results')
        >>> print(f"Output directory: {output_dir}")
    """
    dir_path = Path(directory)
    
    try:
        dir_path.mkdir(parents=parents, exist_ok=exist_ok)
        logger.debug(f"Ensured directory exists: {dir_path}")
        return dir_path
    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        raise


def validate_file_exists(file_path: Union[str, Path], check_readable: bool = True) -> bool:
    """
    Validate that a file exists and is accessible.
    
    Args:
        file_path: Path to file to validate
        check_readable: Also check if file is readable
        
    Returns:
        True if file exists and is accessible
        
    Example:
        >>> if validate_file_exists('data/input.tif'):
        ...     print("File is valid and accessible")
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.debug(f"File does not exist: {path}")
        return False
    
    if not path.is_file():
        logger.debug(f"Path is not a file: {path}")
        return False
    
    if check_readable:
        try:
            with open(path, 'rb') as f:
                f.read(1)  # Try to read first byte
            logger.debug(f"File is readable: {path}")
        except (IOError, OSError) as e:
            logger.debug(f"File is not readable: {path} - {e}")
            return False
    
    return True


def get_file_size(file_path: Union[str, Path], unit: str = 'MB') -> float:
    """
    Get file size in specified unit.
    
    Args:
        file_path: Path to file
        unit: Size unit ('B', 'KB', 'MB', 'GB')
        
    Returns:
        File size in specified unit
        
    Example:
        >>> size_mb = get_file_size('data/large_file.tif', 'MB')
        >>> print(f"File size: {size_mb:.2f} MB")
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    size_bytes = path.stat().st_size
    
    unit_multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4
    }
    
    if unit not in unit_multipliers:
        raise ValueError(f"Unsupported unit: {unit}. Use one of {list(unit_multipliers.keys())}")
    
    size = size_bytes / unit_multipliers[unit]
    logger.debug(f"File size: {path} = {size:.2f} {unit}")
    
    return size


def copy_file(source: Union[str, Path], destination: Union[str, Path], 
              overwrite: bool = False, create_dirs: bool = True) -> Path:
    """
    Copy file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Overwrite destination if it exists
        create_dirs: Create destination directories if needed
        
    Returns:
        Path to copied file
        
    Example:
        >>> copied_file = copy_file('data/input.tif', 'outputs/copied_input.tif')
        >>> print(f"File copied to: {copied_file}")
    """
    src_path = Path(source)
    dst_path = Path(destination)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Destination exists and overwrite=False: {dst_path}")
    
    if create_dirs:
        ensure_directory(dst_path.parent)
    
    try:
        shutil.copy2(src_path, dst_path)
        logger.info(f"Copied file: {src_path} -> {dst_path}")
        return dst_path
    except Exception as e:
        logger.error(f"Failed to copy file: {src_path} -> {dst_path}: {e}")
        raise


def move_file(source: Union[str, Path], destination: Union[str, Path],
              overwrite: bool = False, create_dirs: bool = True) -> Path:
    """
    Move file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Overwrite destination if it exists
        create_dirs: Create destination directories if needed
        
    Returns:
        Path to moved file
        
    Example:
        >>> moved_file = move_file('temp/process_data.tif', 'outputs/final_data.tif')
        >>> print(f"File moved to: {moved_file}")
    """
    src_path = Path(source)
    dst_path = Path(destination)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Destination exists and overwrite=False: {dst_path}")
    
    if create_dirs:
        ensure_directory(dst_path.parent)
    
    try:
        shutil.move(str(src_path), str(dst_path))
        logger.info(f"Moved file: {src_path} -> {dst_path}")
        return dst_path
    except Exception as e:
        logger.error(f"Failed to move file: {src_path} -> {dst_path}: {e}")
        raise


def delete_file_safe(file_path: Union[str, Path], missing_ok: bool = True) -> bool:
    """
    Safely delete a file.
    
    Args:
        file_path: Path to file to delete
        missing_ok: Don't raise error if file doesn't exist
        
    Returns:
        True if file was deleted or didn't exist (with missing_ok=True)
        
    Example:
        >>> success = delete_file_safe('temp/temporary_file.txt')
        >>> print(f"File deletion successful: {success}")
    """
    path = Path(file_path)
    
    if not path.exists():
        if missing_ok:
            logger.debug(f"File already doesn't exist: {path}")
            return True
        else:
            raise FileNotFoundError(f"File not found: {path}")
    
    try:
        path.unlink()
        logger.info(f"Deleted file: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete file: {path}: {e}")
        raise


def create_temp_directory(prefix: str = 'orchestrator_', suffix: str = '', 
                         base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a temporary directory.
    
    Args:
        prefix: Prefix for directory name
        suffix: Suffix for directory name
        base_dir: Base directory for temp directory
        
    Returns:
        Path to created temporary directory
        
    Example:
        >>> temp_dir = create_temp_directory(prefix='processing_')
        >>> print(f"Temporary directory: {temp_dir}")
    """
    base_dir_str = str(base_dir) if base_dir else None
    
    temp_dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=base_dir_str)
    temp_path = Path(temp_dir)
    
    logger.debug(f"Created temporary directory: {temp_path}")
    return temp_path


def find_files(directory: Union[str, Path], pattern: str = '*', 
               recursive: bool = True, file_types: Optional[List[str]] = None) -> List[Path]:
    """
    Find files matching pattern in directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        recursive: Search recursively in subdirectories
        file_types: Filter by file extensions (e.g., ['.tif', '.shp'])
        
    Returns:
        List of matching file paths
        
    Example:
        >>> tif_files = find_files('data', '*.tif', recursive=True)
        >>> print(f"Found {len(tif_files)} TIFF files")
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return []
    
    if recursive:
        search_pattern = dir_path / '**' / pattern
        found_files = list(dir_path.glob(f'**/{pattern}'))
    else:
        found_files = list(dir_path.glob(pattern))
    
    # Filter by file types if specified
    if file_types:
        file_types_lower = [ext.lower() for ext in file_types]
        found_files = [f for f in found_files if f.suffix.lower() in file_types_lower]
    
    # Only return files (not directories)
    found_files = [f for f in found_files if f.is_file()]
    
    logger.debug(f"Found {len(found_files)} files matching pattern '{pattern}' in {dir_path}")
    return found_files


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate hash of file contents.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string
        
    Example:
        >>> file_hash = get_file_hash('data/input.tif', 'sha256')
        >>> print(f"File hash: {file_hash}")
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Select hash algorithm
    if algorithm.lower() == 'md5':
        hasher = hashlib.md5()
    elif algorithm.lower() == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm.lower() == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Calculate hash
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        hash_value = hasher.hexdigest()
        logger.debug(f"File hash ({algorithm}): {path} = {hash_value}")
        return hash_value
        
    except Exception as e:
        logger.error(f"Failed to calculate hash for {path}: {e}")
        raise


def compress_directory(directory: Union[str, Path], output_file: Union[str, Path],
                      format: str = 'zip', exclude_patterns: Optional[List[str]] = None) -> Path:
    """
    Compress directory into archive file.
    
    Args:
        directory: Directory to compress
        output_file: Output archive file path
        format: Archive format ('zip', 'tar', 'tar.gz', 'tar.bz2')
        exclude_patterns: Glob patterns to exclude from archive
        
    Returns:
        Path to created archive file
        
    Example:
        >>> archive = compress_directory('outputs', 'results.zip', format='zip')
        >>> print(f"Archive created: {archive}")
    """
    dir_path = Path(directory)
    output_path = Path(output_file)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Ensure output directory exists
    ensure_directory(output_path.parent)
    
    exclude_patterns = exclude_patterns or []
    
    try:
        if format.lower() == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        # Check exclude patterns
                        skip = False
                        for pattern in exclude_patterns:
                            if file_path.match(pattern):
                                skip = True
                                break
                        
                        if not skip:
                            arcname = file_path.relative_to(dir_path)
                            zf.write(file_path, arcname)
        
        elif format.lower() in ['tar', 'tar.gz', 'tar.bz2']:
            mode_map = {
                'tar': 'w',
                'tar.gz': 'w:gz',
                'tar.bz2': 'w:bz2'
            }
            mode = mode_map[format.lower()]
            
            with tarfile.open(output_path, mode) as tf:
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        # Check exclude patterns
                        skip = False
                        for pattern in exclude_patterns:
                            if file_path.match(pattern):
                                skip = True
                                break
                        
                        if not skip:
                            arcname = file_path.relative_to(dir_path)
                            tf.add(file_path, arcname)
        
        else:
            raise ValueError(f"Unsupported archive format: {format}")
        
        logger.info(f"Compressed directory {dir_path} to {output_path} ({format})")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to compress directory: {e}")
        raise


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
        
    Example:
        >>> info = get_file_info('data/elevation.tif')
        >>> print(f"File size: {info['size_mb']:.2f} MB")
        >>> print(f"Modified: {info['modified_time']}")
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    stat = path.stat()
    
    info = {
        'path': str(path.absolute()),
        'name': path.name,
        'stem': path.stem,
        'suffix': path.suffix,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 ** 2),
        'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'accessed_time': datetime.fromtimestamp(stat.st_atime).isoformat(),
        'is_readable': os.access(path, os.R_OK),
        'is_writable': os.access(path, os.W_OK),
        'is_executable': os.access(path, os.X_OK)
    }
    
    # Add format-specific information
    if path.suffix.lower() in SUPPORTED_RASTER_FORMATS:
        info['file_category'] = 'raster'
    elif path.suffix.lower() in SUPPORTED_VECTOR_FORMATS:
        info['file_category'] = 'vector'
    elif path.suffix.lower() in SUPPORTED_CONFIG_FORMATS:
        info['file_category'] = 'config'
    elif path.suffix.lower() in SUPPORTED_ARCHIVE_FORMATS:
        info['file_category'] = 'archive'
    else:
        info['file_category'] = 'other'
    
    return info


def cleanup_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files.
    
    Args:
        temp_dir: Temporary directory to clean
        max_age_hours: Maximum age of files to keep (in hours)
        
    Returns:
        Number of files deleted
        
    Example:
        >>> deleted_count = cleanup_temp_files('/tmp/orchestrator_temp', max_age_hours=12)
        >>> print(f"Deleted {deleted_count} old temporary files")
    """
    temp_path = Path(temp_dir)
    
    if not temp_path.exists():
        logger.debug(f"Temp directory doesn't exist: {temp_path}")
        return 0
    
    max_age_seconds = max_age_hours * 3600
    current_time = datetime.now().timestamp()
    deleted_count = 0
    
    try:
        for file_path in temp_path.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old temporary files from {temp_path}")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup temp files: {e}")
        return 0


# Export main functions
__all__ = [
    'ensure_directory', 'validate_file_exists', 'get_file_size',
    'copy_file', 'move_file', 'delete_file_safe', 'create_temp_directory',
    'find_files', 'get_file_hash', 'compress_directory', 'get_file_info',
    'cleanup_temp_files'
]
