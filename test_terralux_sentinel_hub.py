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
