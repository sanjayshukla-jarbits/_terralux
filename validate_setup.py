#!/usr/bin/env python3
"""
Validate the complete Sentinel Hub setup
"""
import sys
import subprocess
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'sentinelhub', 'rasterio', 'numpy', 'requests',
        'cryptography', 'keyring'
    ]
    
    print("🐍 Checking Python packages...")
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_file_structure():
    """Check if all required files are present"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        'orchestrator/steps/data_acquisition/real_sentinel_hub_step.py',
        'orchestrator/steps/data_acquisition/configs/sentinel_hub_example.json',
        'orchestrator/steps/data_acquisition/real_impl/auth_manager.py',
        'orchestrator/steps/data_acquisition/real_impl/cache_manager.py',
        'test_sentinel_hub_real.py',
        'test_real_acquisition.py'
    ]
    
    missing = []
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing.append(file_path)
    
    return len(missing) == 0, missing

def check_credentials():
    """Check if credentials are configured"""
    print("\n🔐 Checking credentials...")
    
    import os
    client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
    client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')
    
    if client_id and client_secret:
        print(f"   ✓ Environment variables set")
        print(f"   Client ID: {client_id[:8]}...")
        return True
    else:
        print("   ⚠️  Environment variables not set")
        print("   This is OK for testing with mock data")
        return False

def main():
    """Main validation function"""
    print("🛰️ SENTINEL HUB SETUP VALIDATION")
    print("=" * 40)
    
    all_checks_passed = True
    
    # Check packages
    packages_ok, missing_packages = check_python_packages()
    if not packages_ok:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        all_checks_passed = False
    
    # Check files
    files_ok, missing_files = check_file_structure()
    if not files_ok:
        print(f"\n❌ Missing files: {missing_files}")
        print("Re-run the setup script to create missing files")
        all_checks_passed = False
    
    # Check credentials
    creds_ok = check_credentials()
    
    print(f"\n{'='*40}")
    if all_checks_passed:
        print("✅ Setup validation PASSED!")
        print("\nNext steps:")
        if creds_ok:
            print("1. Run: python test_real_acquisition.py")
        else:
            print("1. Set up credentials:")
            print("   export SENTINEL_HUB_CLIENT_ID='your-client-id'")
            print("   export SENTINEL_HUB_CLIENT_SECRET='your-client-secret'")
            print("2. Run: python test_real_acquisition.py")
        print("3. Integrate with your pipeline!")
    else:
        print("❌ Setup validation FAILED!")
        print("Please fix the issues above and re-run validation")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
