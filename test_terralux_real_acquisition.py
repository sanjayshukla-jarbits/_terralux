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
