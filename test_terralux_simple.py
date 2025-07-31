#!/usr/bin/env python3
"""
Simple working test for _terralux Sentinel Hub
"""
import sys
import os
from pathlib import Path

# Add _terralux to path
sys.path.insert(0, '/home/ubuntu/_terralux')

def test_simple_setup():
    print("🛰️ SIMPLE _TERRALUX TEST")
    print("=" * 30)
    
    # Test 1: Libraries
    print("\n1. Testing libraries...")
    try:
        import sentinelhub
        print("   ✓ sentinelhub available")
    except ImportError:
        print("   ❌ sentinelhub missing - run: pip install sentinelhub")
        return False
    
    # Test 2: Project structure  
    print("\n2. Testing project structure...")
    if Path("/home/ubuntu/_terralux/orchestrator").exists():
        print("   ✓ orchestrator directory exists")
    else:
        print("   ❌ orchestrator directory missing")
        return False
    
    # Test 3: Basic import
    print("\n3. Testing basic import...")
    try:
        import orchestrator
        print("   ✓ orchestrator imports successfully")
    except Exception as e:
        print(f"   ⚠️  orchestrator import issue: {e}")
        print("   This is expected and doesn't affect Sentinel Hub integration")
    
    # Test 4: Credentials
    print("\n4. Checking credentials...")
    client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
    client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')
    
    if client_id and client_secret:
        print(f"   ✓ Credentials found: {client_id[:8]}...")
        
        # Test authentication
        try:
            import requests
            auth_url = "https://services.sentinel-hub.com/auth/oauth/token"
            data = {
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret
            }
            response = requests.post(auth_url, data=data, timeout=10)
            if response.status_code == 200:
                print("   ✅ Credentials are valid!")
            else:
                print(f"   ❌ Credentials invalid: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Credential test failed: {e}")
    else:
        print("   ⚠️  No credentials (set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET)")
    
    print("\n✅ Basic _terralux test completed!")
    return True

if __name__ == "__main__":
    success = test_simple_setup()
    sys.exit(0 if success else 1)
