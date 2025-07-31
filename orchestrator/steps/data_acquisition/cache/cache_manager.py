"""
Cache Manager for _terralux Sentinel Hub Integration
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

class TerraluxCacheManager:
    """Cache manager specifically for _terralux project"""
    
    def __init__(self, cache_dir: Optional[Path] = None, project_root: str = "/home/ubuntu/_terralux"):
        self.project_root = Path(project_root)
        
        # Use _terralux specific cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".terralux_sentinel_hub" / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("Terralux.SentinelHub.Cache")
        
        # Create _terralux cache structure
        (self.cache_dir / "data").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        (self.cache_dir / "requests").mkdir(exist_ok=True)
        (self.cache_dir / "temp").mkdir(exist_ok=True)
        
        self.logger.info(f"✓ Cache initialized for _terralux: {self.cache_dir}")
    
    def generate_cache_key(self, request_params: Dict[str, Any]) -> str:
        """Generate cache key for _terralux project"""
        # Add project identifier to ensure unique keys
        request_params_with_project = {
            **request_params,
            "project": "_terralux",
            "version": "1.0.0"
        }
        
        request_str = json.dumps(request_params_with_project, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()[:12]
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if data is cached in _terralux cache"""
        data_file = self.cache_dir / "data" / f"{cache_key}.tif"
        metadata_file = self.cache_dir / "metadata" / f"{cache_key}.json"
        return data_file.exists() and metadata_file.exists()
    
    def get_cached_data(self, cache_key: str) -> Tuple[Path, Dict[str, Any]]:
        """Get cached data for _terralux project"""
        if not self.is_cached(cache_key):
            raise ValueError(f"Data not cached for key: {cache_key}")
            
        data_file = self.cache_dir / "data" / f"{cache_key}.tif"
        metadata_file = self.cache_dir / "metadata" / f"{cache_key}.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Add cache info for _terralux
        metadata["terralux_cache_info"] = {
            "project": "_terralux",
            "cache_directory": str(self.cache_dir),
            "retrieved_at": datetime.now().isoformat()
        }
        
        self.logger.info(f"✓ Retrieved cached data: {data_file}")
        return data_file, metadata
    
    def cache_data(self, cache_key: str, data_path: Path, metadata: Dict[str, Any]) -> Path:
        """Cache data for _terralux project"""
        cached_data = self.cache_dir / "data" / f"{cache_key}.tif"
        cached_metadata = self.cache_dir / "metadata" / f"{cache_key}.json"
        
        # Copy data to _terralux cache
        shutil.copy2(data_path, cached_data)
        
        # Add _terralux specific metadata
        enhanced_metadata = {
            **metadata,
            "terralux_cache_info": {
                "project": "_terralux",
                "project_root": str(self.project_root),
                "cache_directory": str(self.cache_dir),
                "cached_at": datetime.now().isoformat(),
                "cache_key": cache_key
            }
        }
        
        with open(cached_metadata, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2, default=str)
            
        self.logger.info(f"✓ Data cached for _terralux: {cached_data}")
        return cached_data
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for _terralux project"""
        data_dir = self.cache_dir / "data"
        
        total_size = 0
        file_count = 0
        
        for file_path in data_dir.glob("*.tif"):
            total_size += file_path.stat().st_size
            file_count += 1
            
        return {
            "project": "_terralux",
            "cache_directory": str(self.cache_dir),
            "total_files": file_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "data_directory": str(data_dir),
            "metadata_directory": str(self.cache_dir / "metadata")
        }
