#!/usr/bin/env python3
"""
Test script for Real Sentinel Hub Data Acquisition
"""
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_sentinel_hub_setup():
    """Test the Sentinel Hub setup"""
    print("🧪 TESTING SENTINEL HUB SETUP")
    print("=" * 40)
    
    # Test 1: Check imports
    print("\n1. Testing imports...")
    try:
        import sentinelhub
        print("   ✓ sentinelhub library available")
    except ImportError:
        print("   ❌ sentinelhub library not available")
        return False
    
    try:
        import rasterio
        print("   ✓ rasterio library available")
    except ImportError:
        print("   ❌ rasterio library not available")
        return False
    
    try:
        from orchestrator.steps.data_acquisition.real_impl.auth_manager import SentinelHubAuth
        print("   ✓ Authentication manager available")
    except ImportError as e:
        print(f"   ❌ Authentication manager not available: {e}")
        return False
    
    # Test 2: Configuration
    print("\n2. Testing configuration...")
    config_path = Path("orchestrator/steps/data_acquisition/configs/sentinel_hub_example.json")
    if config_path.exists():
        print("   ✓ Example configuration found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ['credentials', 'endpoints', 'limits']
        for key in required_keys:
            if key in config:
                print(f"   ✓ {key} section present")
            else:
                print(f"   ❌ {key} section missing")
                return False
    else:
        print("   ❌ Example configuration not found")
        return False
    
    # Test 3: Real step class
    print("\n3. Testing real step implementation...")
    try:
        # Import the real step from our created file
        sys.path.append('.')
        from orchestrator.steps.data_acquisition.real_sentinel_hub_step import RealSentinelHubAcquisitionStep
        print("   ✓ Real Sentinel Hub step class available")
        
        # Test step creation
        test_config = {
            'id': 'test_step',
            'type': 'sentinel_hub_acquisition',
            'hyperparameters': {
                'bbox': [85.30, 27.60, 85.32, 27.62],
                'start_date': '2023-06-01',
                'end_date': '2023-06-07',
                'fallback_to_mock': True
            }
        }
        
        step = RealSentinelHubAcquisitionStep(
            test_config['id'],
            test_config['type'],
            test_config['hyperparameters']
        )
        print("   ✓ Step instance created successfully")
        
        # Test execution (should fallback to mock without credentials)
        result = step.execute()
        if result['status'] == 'completed':
            print(f"   ✓ Step execution successful (mock: {result.get('mock', False)})")
        else:
            print(f"   ❌ Step execution failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Real step test failed: {e}")
        return False
    
    print("\n✅ All tests passed! Sentinel Hub setup is ready.")
    print("\nNext steps:")
    print("1. Get your Sentinel Hub credentials from https://apps.sentinel-hub.com/")
    print("2. Set environment variables:")
    print("   export SENTINEL_HUB_CLIENT_ID='your-client-id'")
    print("   export SENTINEL_HUB_CLIENT_SECRET='your-client-secret'")
    print("3. Run: python test_real_acquisition.py")
    
    return True

if __name__ == "__main__":
    success = test_sentinel_hub_setup()
    sys.exit(0 if success else 1)
