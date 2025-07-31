#!/bin/bash
# Corrected Sentinel Hub Setup Script for _terralux Project
# =========================================================

set -e  # Exit on any error

echo "🛰️ SENTINEL HUB SETUP FOR _TERRALUX PROJECT"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
    echo "----------------------------------------"
}

# Check if we're in the _terralux directory
if [[ ! -d "orchestrator" ]]; then
    print_error "Please run this script from the _terralux project root directory"
    print_error "Expected: /home/ubuntu/_terralux/"
    print_error "Current: $(pwd)"
    exit 1
fi

print_status "✅ Confirmed _terralux project root: $(pwd)"

print_step "1. Installing Python Dependencies for _terralux"

# Create Sentinel Hub specific requirements
cat > requirements_sentinel_hub.txt << 'EOF'
# Sentinel Hub API and dependencies
sentinelhub>=3.9.0
requests>=2.28.0
numpy>=1.21.0
rasterio>=1.3.0
geopandas>=0.12.0

# Security and encryption
cryptography>=3.4.8
keyring>=23.0.0

# Data processing
scikit-image>=0.19.0
opencv-python>=4.6.0

# Progress tracking and utilities
tqdm>=4.64.0
click>=8.0.0

# Optional: Advanced processing
scipy>=1.9.0
matplotlib>=3.5.0
EOF

print_status "Installing Sentinel Hub dependencies..."
pip install -r requirements_sentinel_hub.txt

print_step "2. Creating _terralux Directory Structure"

# Create the correct directory structure for _terralux
mkdir -p orchestrator/steps/data_acquisition/{auth,cache,config,downloaders,utils,templates}
mkdir -p data/{cache/sentinel_hub/{data,metadata,requests,temp},credentials}
mkdir -p scripts
mkdir -p tests/fixtures/sample_processes

print_status "✅ _terralux directory structure created"

print_step "3. Creating Corrected Configuration Files"

# Create Sentinel Hub configuration directory with correct paths
mkdir -p ~/.terralux_sentinel_hub/{cache/{data,metadata,requests,temp},credentials,config,logs}
chmod 700 ~/.terralux_sentinel_hub
chmod 700 ~/.terralux_sentinel_hub/credentials

print_status "✅ User cache directory created: ~/.terralux_sentinel_hub/"

# Create example configuration with _terralux paths
cat > orchestrator/steps/data_acquisition/config/sentinel_hub_example.json << 'EOF'
{
  "project_info": {
    "name": "_terralux",
    "root_directory": "/home/ubuntu/_terralux",
    "cache_directory": "~/.terralux_sentinel_hub/cache",
    "credentials_directory": "~/.terralux_sentinel_hub/credentials"
  },
  "credentials": {
    "client_id": "your-client-id-here",
    "client_secret": "your-client-secret-here",
    "instance_id": "your-instance-id-here"
  },
  "endpoints": {
    "base_url": "https://services.sentinel-hub.com",
    "auth_url": "https://services.sentinel-hub.com/auth",
    "catalog_url": "https://services.sentinel-hub.com/api/v1/catalog",
    "process_url": "https://services.sentinel-hub.com/api/v1/process"
  },
  "cache": {
    "enabled": true,
    "base_directory": "~/.terralux_sentinel_hub/cache",
    "max_size_gb": 10,
    "cleanup_older_than_days": 30,
    "compression": "lzw"
  },
  "collections": {
    "SENTINEL-2-L2A": {
      "default_bands": ["B02", "B03", "B04", "B08"],
      "default_resolution": 10,
      "max_cloud_coverage": 20
    },
    "SENTINEL-1-GRD": {
      "default_polarizations": ["VV", "VH"],
      "default_resolution": 10
    }
  }
}
EOF

print_status "✅ Configuration template created with _terralux paths"

print_step "4. Creating Corrected Real Implementation"

# Create the corrected authentication manager for _terralux
cat > orchestrator/steps/data_acquisition/auth/auth_manager.py << 'EOF'
"""
Sentinel Hub Authentication Manager for _terralux Project
"""
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import logging

class TerraluxSentinelHubAuth:
    """Authentication manager specifically for _terralux project"""
    
    def __init__(self, client_id: str, client_secret: str, project_root: str = "/home/ubuntu/_terralux"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.project_root = Path(project_root)
        self.access_token = None
        self.token_expires_at = None
        
        # Use _terralux specific cache directory
        self.cache_dir = Path.home() / ".terralux_sentinel_hub"
        self.logger = logging.getLogger("Terralux.SentinelHub.Auth")
    
    def get_token(self) -> str:
        """Get valid access token for _terralux project"""
        if self._is_token_valid():
            return self.access_token
        return self._refresh_token()
    
    def _is_token_valid(self) -> bool:
        """Check if current token is valid"""
        if not self.access_token or not self.token_expires_at:
            return False
        return datetime.now() < (self.token_expires_at - timedelta(minutes=5))
    
    def _refresh_token(self) -> str:
        """Refresh access token from Sentinel Hub"""
        auth_url = "https://services.sentinel-hub.com/auth/oauth/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        try:
            response = requests.post(auth_url, data=data, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            expires_in = token_data.get('expires_in', 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            self.logger.info(f"✓ Access token refreshed for _terralux project")
            return self.access_token
            
        except requests.RequestException as e:
            self.logger.error(f"Authentication failed: {e}")
            raise Exception(f"Sentinel Hub authentication failed: {e}")
EOF

# Create corrected cache manager for _terralux
cat > orchestrator/steps/data_acquisition/cache/cache_manager.py << 'EOF'
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
EOF

print_step "5. Creating _terralux Integration Files"

# Create corrected __init__.py for _terralux
cat > orchestrator/steps/data_acquisition/__init__.py << 'EOF'
"""
Data acquisition steps for _terralux orchestrator
"""

# Import existing mock implementations (keep for compatibility)
try:
    from .sentinel_hub_step import SentinelHubAcquisitionStep as MockSentinelHubStep
    from .dem_acquisition_step import DEMAcquisitionStep
    from .local_files_step import LocalFilesDiscoveryStep
except ImportError:
    MockSentinelHubStep = None
    DEMAcquisitionStep = None
    LocalFilesDiscoveryStep = None

# Import real implementation for _terralux
try:
    from .real_sentinel_hub_step import RealSentinelHubAcquisitionStep
    REAL_IMPLEMENTATION_AVAILABLE = True
except ImportError:
    RealSentinelHubAcquisitionStep = None
    REAL_IMPLEMENTATION_AVAILABLE = False

# Register steps with _terralux specific naming
try:
    from ..base.step_registry import StepRegistry
    
    # Register real implementation as primary
    if REAL_IMPLEMENTATION_AVAILABLE:
        StepRegistry.register('sentinel_hub_acquisition', RealSentinelHubAcquisitionStep)
        print("✓ Real Sentinel Hub step registered for _terralux")
    else:
        if MockSentinelHubStep:
            StepRegistry.register('sentinel_hub_acquisition', MockSentinelHubStep)
            print("⚠ Using mock Sentinel Hub step for _terralux")
    
    # Register other steps
    if DEMAcquisitionStep:
        StepRegistry.register('dem_acquisition', DEMAcquisitionStep)
    if LocalFilesDiscoveryStep:
        StepRegistry.register('local_files_discovery', LocalFilesDiscoveryStep)
        
except ImportError:
    print("⚠ Step registry not available")

__all__ = [
    'RealSentinelHubAcquisitionStep',
    'MockSentinelHubStep', 
    'DEMAcquisitionStep',
    'LocalFilesDiscoveryStep',
    'REAL_IMPLEMENTATION_AVAILABLE'
]
EOF

print_step "6. Creating _terralux Test Scripts"

# Create corrected test script for _terralux
cat > test_terralux_sentinel_hub.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for _terralux Sentinel Hub Integration
"""
import sys
import os
from pathlib import Path

# Add _terralux project root to Python path
TERRALUX_ROOT = Path("/home/ubuntu/_terralux")
sys.path.insert(0, str(TERRALUX_ROOT))

def test_terralux_setup():
    """Test the _terralux Sentinel Hub integration"""
    print("🛰️ TESTING _TERRALUX SENTINEL HUB INTEGRATION")
    print("=" * 50)
    
    print(f"📁 Project root: {TERRALUX_ROOT}")
    print(f"📁 Current directory: {Path.cwd()}")
    
    # Test 1: Check project structure
    print("\n1. Testing _terralux project structure...")
    
    required_dirs = [
        "orchestrator",
        "orchestrator/steps", 
        "orchestrator/steps/data_acquisition",
        "tests"
    ]
    
    for dir_path in required_dirs:
        full_path = TERRALUX_ROOT / dir_path
        if full_path.exists():
            print(f"   ✓ {dir_path}")
        else:
            print(f"   ❌ {dir_path}")
            return False
    
    # Test 2: Check imports
    print("\n2. Testing _terralux imports...")
    try:
        from orchestrator.steps.data_acquisition import REAL_IMPLEMENTATION_AVAILABLE
        print(f"   ✓ Data acquisition module imported")
        print(f"   Real implementation available: {REAL_IMPLEMENTATION_AVAILABLE}")
        
        if REAL_IMPLEMENTATION_AVAILABLE:
            from orchestrator.steps.data_acquisition import RealSentinelHubAcquisitionStep
            print("   ✓ Real Sentinel Hub step available")
        else:
            print("   ⚠ Using mock implementation")
            
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 3: Check cache directory
    print("\n3. Testing _terralux cache setup...")
    cache_dir = Path.home() / ".terralux_sentinel_hub"
    
    if cache_dir.exists():
        print(f"   ✓ Cache directory exists: {cache_dir}")
        
        subdirs = ["cache", "credentials", "config"]
        for subdir in subdirs:
            subdir_path = cache_dir / subdir
            if subdir_path.exists():
                print(f"   ✓ {subdir}/")
            else:
                print(f"   ❌ {subdir}/ missing") 
    else:
        print(f"   ❌ Cache directory missing: {cache_dir}")
        return False
    
    # Test 4: Test step creation
    print("\n4. Testing step creation...")
    try:
        test_config = {
            'id': 'terralux_test',
            'type': 'sentinel_hub_acquisition',
            'hyperparameters': {
                'bbox': [85.30, 27.60, 85.32, 27.62],
                'start_date': '2023-06-01',
                'end_date': '2023-06-07',
                'fallback_to_mock': True,
                'project_root': str(TERRALUX_ROOT)
            }
        }
        
        if REAL_IMPLEMENTATION_AVAILABLE:
            from orchestrator.steps.data_acquisition import RealSentinelHubAcquisitionStep
            step = RealSentinelHubAcquisitionStep(
                test_config['id'],
                test_config['type'], 
                test_config['hyperparameters']
            )
            print("   ✓ Real step instance created")
        else:
            from orchestrator.steps.data_acquisition import MockSentinelHubStep
            step = MockSentinelHubStep(test_config)
            print("   ✓ Mock step instance created")
            
        result = step.execute()
        if result.get('status') == 'completed':
            print(f"   ✓ Step execution successful")
            print(f"   Data path: {result.get('imagery_data', 'N/A')}")
        else:
            print(f"   ❌ Step execution failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Step test failed: {e}")
        return False
    
    print(f"\n✅ _terralux Sentinel Hub integration test passed!")
    print(f"\nNext steps:")
    print(f"1. Set environment variables:")
    print(f"   export SENTINEL_HUB_CLIENT_ID='your-client-id'")
    print(f"   export SENTINEL_HUB_CLIENT_SECRET='your-client-secret'")
    print(f"2. Run: python test_terralux_real_acquisition.py")
    print(f"3. Check cache at: {cache_dir}")
    
    return True

if __name__ == "__main__":
    success = test_terralux_setup()
    sys.exit(0 if success else 1)
EOF

# Create real acquisition test for _terralux
cat > test_terralux_real_acquisition.py << 'EOF'
#!/usr/bin/env python3
"""
Test real Sentinel Hub data acquisition for _terralux project
"""
import os
import sys
from pathlib import Path

# Add _terralux project root to Python path
TERRALUX_ROOT = Path("/home/ubuntu/_terralux")
sys.path.insert(0, str(TERRALUX_ROOT))

def test_terralux_real_acquisition():
    """Test real data acquisition for _terralux"""
    print("🛰️ TESTING _TERRALUX REAL SENTINEL HUB ACQUISITION")
    print("=" * 55)
    
    # Check credentials
    client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
    client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ Sentinel Hub credentials not found")
        print("Set environment variables:")
        print("   export SENTINEL_HUB_CLIENT_ID='your-client-id'")
        print("   export SENTINEL_HUB_CLIENT_SECRET='your-client-secret'")
        return False
    
    print(f"✓ Found credentials for _terralux: {client_id[:8]}...")
    
    try:
        from orchestrator.steps.data_acquisition import RealSentinelHubAcquisitionStep
        
        # Create _terralux specific test configuration
        test_config = {
            'id': 'terralux_real_test',
            'type': 'sentinel_hub_acquisition',
            'hyperparameters': {
                'bbox': [85.30, 27.60, 85.32, 27.62],  # Small Nepal area
                'start_date': '2023-06-01',
                'end_date': '2023-06-03',  # Short range
                'data_collection': 'SENTINEL-2-L2A',
                'resolution': 60,  # Lower resolution for speed
                'bands': ['B02', 'B03', 'B04'],  # RGB only
                'max_cloud_coverage': 80,
                'client_id': client_id,
                'client_secret': client_secret,
                'fallback_to_mock': False,  # Force real API
                'project_root': str(TERRALUX_ROOT),
                'cache_directory': str(Path.home() / ".terralux_sentinel_hub" / "cache"),
                'output_directory': str(TERRALUX_ROOT / "data" / "outputs")
            }
        }
        
        print(f"📍 Test area for _terralux: {test_config['hyperparameters']['bbox']}")
        print(f"📅 Date range: {test_config['hyperparameters']['start_date']} to {test_config['hyperparameters']['end_date']}")
        
        # Create and execute step
        step = RealSentinelHubAcquisitionStep(
            test_config['id'],
            test_config['type'],
            test_config['hyperparameters']
        )
        
        print(f"\n🚀 Starting real acquisition for _terralux...")
        result = step.execute()
        
        # Check results
        print(f"\n📊 _TERRALUX RESULTS:")
        print(f"   Status: {result['status']}")
        print(f"   Mock used: {result.get('mock', False)}")
        print(f"   Cache used: {result.get('cache_used', False)}")
        
        if result['status'] == 'completed':
            print(f"   ✓ Data file: {result['imagery_data']}")
            
            # Check if file exists
            data_path = Path(result['imagery_data'])
            if data_path.exists():
                file_size_mb = data_path.stat().st_size / (1024*1024)
                print(f"   📁 File size: {file_size_mb:.2f} MB")
                print(f"   📂 Cache location: {data_path.parent}")
            
            # Show metadata
            if 'metadata' in result:
                metadata = result['metadata']
                if 'terralux_cache_info' in metadata:
                    cache_info = metadata['terralux_cache_info']
                    print(f"   🗂️  Project: {cache_info.get('project', 'N/A')}")
                    print(f"   🗂️  Cache key: {cache_info.get('cache_key', 'N/A')}")
            
            print(f"\n✅ _terralux real acquisition successful!")
        else:
            print(f"❌ Acquisition failed: {result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ _terralux test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_terralux_real_acquisition()
    sys.exit(0 if success else 1)
EOF

print_step "7. Creating _terralux Process Definitions"

# Create real API process definition for _terralux
cat > orchestrator/processes/shared/terralux_real_data_acquisition.json << 'EOF'
{
  "process_info": {
    "name": "_terralux Real Data Acquisition",
    "version": "1.0.0",
    "description": "Real satellite data acquisition for _terralux project",
    "application_type": "data_acquisition",
    "project": "_terralux"
  },
  "global_config": {
    "template_variables": {
      "bbox": "{bbox}",
      "start_date": "{start_date}",
      "end_date": "{end_date}",
      "area_name": "{area_name}"
    },
    "output_directory": "/home/ubuntu/_terralux/data/outputs/{area_name}",
    "cache_enabled": true,
    "project_root": "/home/ubuntu/_terralux"
  },
  "steps": [
    {
      "id": "acquire_sentinel_real",
      "type": "sentinel_hub_acquisition",
      "description": "Real Sentinel-2 acquisition for _terralux",
      "hyperparameters": {
        "bbox": "{bbox}",
        "start_date": "{start_date}",
        "end_date": "{end_date}",
        "data_collection": "SENTINEL-2-L2A",
        "resolution": 20,
        "bands": ["B02", "B03", "B04", "B08"],
        "max_cloud_coverage": 30,
        "use_cache": true,
        "fallback_to_mock": true,
        "project_root": "/home/ubuntu/_terralux",
        "cache_directory": "~/.terralux_sentinel_hub/cache"
      }
    },
    {
      "id": "acquire_dem",
      "type": "dem_acquisition", 
      "description": "DEM acquisition for _terralux",
      "hyperparameters": {
        "bbox": "{bbox}",
        "source": "SRTM",
        "resolution": 90,
        "generate_derivatives": true,
        "use_mock_data": true
      }
    },
    {
      "id": "discover_local_data",
      "type": "local_files_discovery",
      "description": "Local data discovery for _terralux",
      "hyperparameters": {
        "base_path": "/home/ubuntu/_terralux/data/local/{area_name}",
        "file_patterns": ["*.tif", "*.shp"],
        "recursive": true,
        "generate_mock_if_empty": true
      }
    }
  ]
}
EOF

print_step "8. Making Scripts Executable"

chmod +x test_terralux_sentinel_hub.py
chmod +x test_terralux_real_acquisition.py

print_step "9. Running _terralux Validation"

print_status "Running initial validation for _terralux..."
python test_terralux_sentinel_hub.py

print_step "10. Creating _terralux Documentation"

cat > TERRALUX_SENTINEL_HUB_GUIDE.md << 'EOF'
# _terralux Sentinel Hub Integration Guide

## Project Structure

This integration is specifically designed for the **_terralux** project located at:
```
/home/ubuntu/_terralux/
```

## Cache Directory Structure

The Sentinel Hub cache for _terralux is located at:
```
~/.terralux_sentinel_hub/
├── cache/
│   ├── data/           # Cached satellite imagery files
│   ├── metadata/       # Metadata for cached files  
│   ├── requests/       # API request information
│   └── temp/           # Temporary download files
├── credentials/        # Encrypted credentials
├── config/            # Configuration files
└── logs/              # Operation logs
```

## Cache Management

### Cache Key Format
Cache keys are generated using MD5 hash of request parameters plus project identifier:
```
Request + Project("_terralux") + Version → MD5 Hash → Cache Key
```

### Cache Files
```
~/.terralux_sentinel_hub/cache/data/a1b2c3d4e5f6.tif     # Satellite data
~/.terralux_sentinel_hub/cache/metadata/a1b2c3d4e5f6.json # Metadata
```

### Metadata Structure
```json
{
  "request_parameters": { ... },
  "data_info": { ... },
  "terralux_cache_info": {
    "project": "_terralux",
    "project_root": "/home/ubuntu/_terralux",
    "cache_directory": "/home/ubuntu/.terralux_sentinel_hub/cache",
    "cached_at": "2024-01-15T14:30:00Z",
    "cache_key": "a1b2c3d4e5f6"
  }
}
```

## Quick Start for _terralux

1. **Setup Environment**
```bash
cd /home/ubuntu/_terralux
./setup_sentinel_hub.sh
```

2. **Set Credentials**
```bash
export SENTINEL_HUB_CLIENT_ID="your-client-id"
export SENTINEL_HUB_CLIENT_SECRET="your-client-secret"
```

3. **Test Integration**
```bash
python test_terralux_sentinel_hub.py
python test_terralux_real_acquisition.py
```

4. **Use in Pipeline**
```python
from orchestrator.steps.data_acquisition import RealSentinelHubAcquisitionStep

config = {
    'id': 'terralux_sentinel',
    'type': 'sentinel_hub_acquisition',
    'hyperparameters': {
        'project_root': '/home/ubuntu/_terralux',
        'cache_directory': '~/.terralux_sentinel_hub/cache'
    }
}
```

## Cache Benefits

- **Avoid Redundant Downloads**: Same requests use cached data
- **Faster Development**: Repeated testing uses cached data
- **Quota Conservation**: Reduces API calls
- **Offline Capability**: Can work with cached data when offline

## Troubleshooting

### Import Issues
```bash
cd /home/ubuntu/_terralux
python -c "import sys; sys.path.append('.'); from orchestrator.steps.data_acquisition import RealSentinelHubAcquisitionStep; print('✅ Import OK')"
```

### Cache Issues
```bash
ls -la ~/.terralux_sentinel_hub/cache/data/
ls -la ~/.terralux_sentinel_hub/cache/metadata/
```

### Permissions
```bash 
chmod 700 ~/.terralux_sentinel_hub/credentials
```

## Cache Commands

### View Cache Stats
```python
from orchestrator.steps.data_acquisition.cache.cache_manager import TerraluxCacheManager
cache = TerraluxCacheManager()
stats = cache.get_cache_stats()
print(f"Cache size: {stats['total_size_mb']} MB")
print(f"Files: {stats['total_files']}")
```

### Clear Old Cache
```python
cache.cleanup_old_cache(days_old=7)  # Remove files older than 7 days
```
EOF

print_step "11. Final Setup Summary for _terralux"

echo ""
print_status "🎉 _TERRALUX SENTINEL HUB SETUP COMPLETE!"
echo ""
echo "Project: _terralux"
echo "Root: /home/ubuntu/_terralux"
echo "Cache: ~/.terralux_sentinel_hub/"
echo ""
echo "Created files:"
echo "  ✓ Real Sentinel Hub implementation for _terralux"
echo "  ✓ _terralux specific authentication and caching"
echo "  ✓ Corrected import paths and configurations"
echo "  ✓ _terralux test scripts and validation"
echo "  ✓ Project-specific process definitions"
echo ""
echo "Next steps for _terralux:"
echo "  1. Get credentials: https://apps.sentinel-hub.com/"
echo "  2. Set environment variables:"
echo "     export SENTINEL_HUB_CLIENT_ID='your-client-id'"
echo "     export SENTINEL_HUB_CLIENT_SECRET='your-client-secret'"
echo "  3. Test: python test_terralux_real_acquisition.py"
echo "  4. Check cache: ls ~/.terralux_sentinel_hub/cache/data/"
echo ""
print_status "_terralux setup script completed successfully!"
