#!/usr/bin/env python3
"""
Test real Sentinel Hub data acquisition with actual API calls
"""
import os
import sys
import json
import tempfile
from pathlib import Path

def test_real_acquisition():
    """Test real data acquisition"""
    print("🛰️ TESTING REAL SENTINEL HUB DATA ACQUISITION")
    print("=" * 50)
    
    # Check credentials
    client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
    client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ Sentinel Hub credentials not found in environment variables")
        print("Please set:")
        print("  export SENTINEL_HUB_CLIENT_ID='your-client-id'")
        print("  export SENTINEL_HUB_CLIENT_SECRET='your-client-secret'")
        return False
    
    print(f"✓ Found credentials for client: {client_id[:8]}...")
    
    # Import and test real step
    try:
        sys.path.append('.')
        from orchestrator.steps.data_acquisition.real_sentinel_hub_step import RealSentinelHubAcquisitionStep
        
        # Create test configuration
        test_config = {
            'id': 'real_test',
            'type': 'sentinel_hub_acquisition',
            'hyperparameters': {
                'bbox': [85.30, 27.60, 85.32, 27.62],  # Small area in Nepal
                'start_date': '2023-06-01',
                'end_date': '2023-06-03',  # Short time range
                'data_collection': 'SENTINEL-2-L2A',
                'resolution': 60,  # Lower resolution for faster download
                'bands': ['B02', 'B03', 'B04'],  # Just RGB bands
                'max_cloud_coverage': 80,  # Allow high cloud coverage
                'client_id': client_id,
                'client_secret': client_secret,
                'fallback_to_mock': False,  # Force real acquisition
                'cache_directory': tempfile.mkdtemp(prefix='sentinel_test_'),
                'output_directory': tempfile.mkdtemp(prefix='sentinel_output_')
            }
        }
        
        print(f"📍 Test area: {test_config['hyperparameters']['bbox']}")
        print(f"📅 Date range: {test_config['hyperparameters']['start_date']} to {test_config['hyperparameters']['end_date']}")
        print(f"🎯 Collection: {test_config['hyperparameters']['data_collection']}")
        
        # Create and execute step
        step = RealSentinelHubAcquisitionStep(
            test_config['id'],
            test_config['type'],
            test_config['hyperparameters']
        )
        
        print("\n🚀 Starting real data acquisition...")
        result = step.execute()
        
        # Check results
        print(f"\n📊 RESULTS:")
        print(f"   Status: {result['status']}")
        print(f"   Mock used: {result.get('mock', False)}")
        print(f"   Cache used: {result.get('cache_used', False)}")
        
        if result['status'] == 'completed':
            print(f"   ✓ Data file: {result['imagery_data']}")
            
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"   📏 Data shape: {metadata.get('data_shape', 'N/A')}")
                print(f"   💾 File size: {metadata.get('file_size_mb', 0):.2f} MB")
                
                if 'statistics' in metadata:
                    stats = metadata['statistics']
                    print(f"   📈 Data range: {stats['min']:.3f} to {stats['max']:.3f}")
            
            print("\n✅ Real data acquisition successful!")
            
            # Verify file exists
            data_path = Path(result['imagery_data'])
            if data_path.exists():
                print(f"✓ Output file exists: {data_path}")
                print(f"  File size: {data_path.stat().st_size / (1024*1024):.2f} MB")
            else:
                print("❌ Output file not found")
                return False
                
        else:
            print(f"❌ Acquisition failed: {result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_acquisition()
    sys.exit(0 if success else 1)
