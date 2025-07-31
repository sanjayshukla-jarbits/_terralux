"""
Data Cache Manager for Sentinel Hub
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

class DataCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".sentinel_hub_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("SentinelHub.Cache")
        
        # Create cache structure
        (self.cache_dir / "data").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
    
    def generate_key(self, request_params: Dict[str, Any]) -> str:
        """Generate cache key from request parameters"""
        request_str = json.dumps(request_params, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if data is cached"""
        data_file = self.cache_dir / "data" / f"{cache_key}.tif"
        return data_file.exists()
    
    def get_cached_data(self, cache_key: str) -> Tuple[Path, Dict[str, Any]]:
        """Get cached data and metadata"""
        data_file = self.cache_dir / "data" / f"{cache_key}.tif"
        metadata_file = self.cache_dir / "metadata" / f"{cache_key}.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return data_file, metadata
    
    def cache_data(self, cache_key: str, data_path: Path, metadata: Dict[str, Any]) -> Path:
        """Cache data and metadata"""
        cached_data = self.cache_dir / "data" / f"{cache_key}.tif"
        cached_metadata = self.cache_dir / "metadata" / f"{cache_key}.json"
        
        shutil.copy2(data_path, cached_data)
        
        with open(cached_metadata, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return cached_data
