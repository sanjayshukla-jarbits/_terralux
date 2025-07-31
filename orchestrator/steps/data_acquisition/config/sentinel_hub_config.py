#!/usr/bin/env python3
"""
Sentinel Hub Configuration Manager
=================================

Manages configuration, credentials, and settings for Sentinel Hub API access.
Provides secure credential handling and configuration validation.

Features:
- Multiple credential source support (env vars, config files, parameters)
- Secure credential storage and retrieval
- Configuration validation and testing
- Support for multiple Sentinel Hub instances
- Configuration templates and examples

Author: Pipeline Development Team
Version: 1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import keyring
from cryptography.fernet import Fernet
import base64


@dataclass
class SentinelHubCredentials:
    """Sentinel Hub API credentials"""
    client_id: str
    client_secret: str
    instance_id: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if credentials are complete"""
        return bool(self.client_id and self.client_secret)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary (excluding None values)"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SentinelHubEndpoints:
    """Sentinel Hub API endpoints"""
    base_url: str = "https://services.sentinel-hub.com"
    auth_url: str = "https://services.sentinel-hub.com/auth"
    catalog_url: str = "https://services.sentinel-hub.com/api/v1/catalog"
    statistical_url: str = "https://services.sentinel-hub.com/api/v1/statistics"
    process_url: str = "https://services.sentinel-hub.com/api/v1/process"
    
    def validate_endpoints(self) -> bool:
        """Validate endpoint URLs"""
        import requests
        try:
            # Test base URL connectivity
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False


@dataclass
class SentinelHubLimits:
    """API rate limits and quotas"""
    max_requests_per_minute: int = 60
    max_processing_units_per_month: int = 1000
    max_concurrent_requests: int = 4
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: int = 2


class SentinelHubConfigManager:
    """Manages Sentinel Hub configuration and credentials"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / ".sentinel_hub"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "config.json"
        self.credentials_file = self.config_dir / "credentials.json"
        self.cache_dir = self.config_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("SentinelHub.Config")
        
        # Initialize encryption for secure credential storage
        self.encryption_key = self._get_or_create_encryption_key()
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for secure credential storage"""
        key_file = self.config_dir / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Make key file readable only by owner
            key_file.chmod(0o600)
            return key
    
    def save_credentials(self, credentials: SentinelHubCredentials, encrypt: bool = True) -> bool:
        """Save credentials securely"""
        try:
            cred_data = credentials.to_dict()
            
            if encrypt:
                # Encrypt sensitive data
                fernet = Fernet(self.encryption_key)
                cred_data['client_secret'] = fernet.encrypt(
                    cred_data['client_secret'].encode()
                ).decode()
                cred_data['_encrypted'] = True
            
            with open(self.credentials_file, 'w') as f:
                json.dump(cred_data, f, indent=2)
            
            # Set secure file permissions
            self.credentials_file.chmod(0o600)
            
            self.logger.info("✓ Credentials saved securely")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save credentials: {e}")
            return False
    
    def load_credentials(self) -> Optional[SentinelHubCredentials]:
        """Load credentials from various sources"""
        # Try loading from file first
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, 'r') as f:
                    cred_data = json.load(f)
                
                # Decrypt if encrypted
                if cred_data.get('_encrypted', False):
                    fernet = Fernet(self.encryption_key)
                    cred_data['client_secret'] = fernet.decrypt(
                        cred_data['client_secret'].encode()
                    ).decode()
                    del cred_data['_encrypted']
                
                return SentinelHubCredentials(**cred_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to load credentials from file: {e}")
        
        # Try environment variables
        env_credentials = self._load_from_environment()
        if env_credentials and env_credentials.is_valid():
            return env_credentials
        
        # Try system keyring
        keyring_credentials = self._load_from_keyring()
        if keyring_credentials and keyring_credentials.is_valid():
            return keyring_credentials
        
        self.logger.warning("No valid credentials found")
        return None
    
    def _load_from_environment(self) -> Optional[SentinelHubCredentials]:
        """Load credentials from environment variables"""
        client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
        client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')
        instance_id = os.getenv('SENTINEL_HUB_INSTANCE_ID')
        
        if client_id and client_secret:
            return SentinelHubCredentials(
                client_id=client_id,
                client_secret=client_secret,
                instance_id=instance_id
            )
        return None
    
    def _load_from_keyring(self) -> Optional[SentinelHubCredentials]:
        """Load credentials from system keyring"""
        try:
            client_id = keyring.get_password("sentinel_hub", "client_id")
            client_secret = keyring.get_password("sentinel_hub", "client_secret")
            instance_id = keyring.get_password("sentinel_hub", "instance_id")
            
            if client_id and client_secret:
                return SentinelHubCredentials(
                    client_id=client_id,
                    client_secret=client_secret,
                    instance_id=instance_id
                )
        except Exception as e:
            self.logger.debug(f"Keyring access failed: {e}")
        
        return None
    
    def save_to_keyring(self, credentials: SentinelHubCredentials) -> bool:
        """Save credentials to system keyring"""
        try:
            keyring.set_password("sentinel_hub", "client_id", credentials.client_id)
            keyring.set_password("sentinel_hub", "client_secret", credentials.client_secret)
            if credentials.instance_id:
                keyring.set_password("sentinel_hub", "instance_id", credentials.instance_id)
            
            self.logger.info("✓ Credentials saved to system keyring")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save to keyring: {e}")
            return False
    
    def test_credentials(self, credentials: Optional[SentinelHubCredentials] = None) -> Dict[str, Any]:
        """Test credentials by making an authentication request"""
        if not credentials:
            credentials = self.load_credentials()
        
        if not credentials or not credentials.is_valid():
            return {
                'valid': False,
                'error': 'No valid credentials found'
            }
        
        try:
            import requests
            
            auth_url = "https://services.sentinel-hub.com/auth/oauth/token"
            data = {
                'grant_type': 'client_credentials',
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret
            }
            
            response = requests.post(auth_url, data=data, timeout=30)
            
            if response.status_code == 200:
                token_data = response.json()
                return {
                    'valid': True,
                    'token_type': token_data.get('token_type'),
                    'expires_in': token_data.get('expires_in'),
                    'account_id': token_data.get('sub')
                }
            else:
                return {
                    'valid': False,
                    'error': f'Authentication failed: {response.status_code}',
                    'details': response.text
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': f'Connection failed: {str(e)}'
            }
    
    def create_config_template(self) -> Dict[str, Any]:
        """Create a configuration template"""
        return {
            "credentials": {
                "client_id": "your-client-id-here",
                "client_secret": "your-client-secret-here",
                "instance_id": "your-instance-id-here (optional)"
            },
            "endpoints": asdict(SentinelHubEndpoints()),
            "limits": asdict(SentinelHubLimits()),
            "cache": {
                "enabled": True,
                "directory": str(self.cache_dir),
                "max_size_gb": 10,
                "cleanup_older_than_days": 30
            },
            "download": {
                "chunk_size_mb": 10,
                "max_retries": 3,
                "timeout_seconds": 300,
                "parallel_downloads": 4
            },
            "data_collections": {
                "sentinel1": {
                    "default_polarizations": ["VV", "VH"],
                    "default_resolution": 10,
                    "orbit_direction": "ASCENDING"
                },
                "sentinel2": {
                    "default_bands": ["B02", "B03", "B04", "B08"],
                    "default_resolution": 10,
                    "max_cloud_coverage": 20
                }
            }
        }
    
    def save_config_template(self, file_path: Optional[Path] = None) -> Path:
        """Save configuration template to file"""
        if not file_path:
            file_path = self.config_dir / "config_template.json"
        
        template = self.create_config_template()
        
        with open(file_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        self.logger.info(f"✓ Configuration template saved to {file_path}")
        return file_path
    
    def setup_interactive(self) -> bool:
        """Interactive setup for first-time configuration"""
        print("🛰️ Sentinel Hub Configuration Setup")
        print("=" * 40)
        
        # Get credentials
        print("\n📋 Enter your Sentinel Hub credentials:")
        print("(You can get these from https://apps.sentinel-hub.com/)")
        
        client_id = input("Client ID: ").strip()
        client_secret = input("Client Secret: ").strip()
        instance_id = input("Instance ID (optional): ").strip() or None
        
        if not client_id or not client_secret:
            print("❌ Client ID and Client Secret are required")
            return False
        
        # Create credentials
        credentials = SentinelHubCredentials(
            client_id=client_id,
            client_secret=client_secret,
            instance_id=instance_id
        )
        
        # Test credentials
        print("\n🔍 Testing credentials...")
        test_result = self.test_credentials(credentials)
        
        if not test_result['valid']:
            print(f"❌ Credentials test failed: {test_result['error']}")
            return False
        
        print("✅ Credentials are valid!")
        
        # Save credentials
        save_options = {
            '1': ('File (encrypted)', lambda: self.save_credentials(credentials, encrypt=True)),
            '2': ('System keyring', lambda: self.save_to_keyring(credentials)),
            '3': ('Both', lambda: self.save_credentials(credentials) and self.save_to_keyring(credentials))
        }
        
        print("\n💾 How would you like to save your credentials?")
        for key, (desc, _) in save_options.items():
            print(f"{key}. {desc}")
        
        choice = input("Choose (1-3): ").strip()
        
        if choice in save_options:
            desc, save_func = save_options[choice]
            if save_func():
                print(f"✅ Credentials saved using: {desc}")
            else:
                print("❌ Failed to save credentials")
                return False
        else:
            print("❌ Invalid choice")
            return False
        
        # Save configuration template
        print("\n📄 Creating configuration template...")
        template_path = self.save_config_template()
        print(f"✅ Template saved to: {template_path}")
        print("You can edit this file to customize your configuration.")
        
        print(f"\n🎉 Setup complete! Configuration saved to: {self.config_dir}")
        return True
    
    def get_quota_usage(self, credentials: Optional[SentinelHubCredentials] = None) -> Dict[str, Any]:
        """Get current quota usage from Sentinel Hub"""
        if not credentials:
            credentials = self.load_credentials()
        
        if not credentials:
            return {'error': 'No credentials available'}
        
        try:
            # This would require additional API calls to get quota information
            # Implementation depends on Sentinel Hub's quota API
            return {
                'processing_units_used': 0,  # Placeholder
                'processing_units_limit': 1000,  # Placeholder
                'requests_today': 0,  # Placeholder
                'last_updated': 'Not implemented'
            }
        except Exception as e:
            return {'error': str(e)}


class SentinelHubConfigValidator:
    """Validates Sentinel Hub configuration"""
    
    @staticmethod
    def validate_credentials(credentials: SentinelHubCredentials) -> List[str]:
        """Validate credentials format"""
        errors = []
        
        if not credentials.client_id:
            errors.append("Client ID is required")
        elif len(credentials.client_id) < 10:
            errors.append("Client ID appears to be too short")
        
        if not credentials.client_secret:
            errors.append("Client Secret is required")
        elif len(credentials.client_secret) < 20:
            errors.append("Client Secret appears to be too short")
        
        if credentials.instance_id and len(credentials.instance_id) < 10:
            errors.append("Instance ID appears to be invalid")
        
        return errors
    
    @staticmethod
    def validate_bbox(bbox: List[float]) -> List[str]:
        """Validate bounding box coordinates"""
        errors = []
        
        if len(bbox) != 4:
            errors.append("Bounding box must have exactly 4 coordinates")
            return errors
        
        west, south, east, north = bbox
        
        if not (-180 <= west <= 180):
            errors.append("West coordinate must be between -180 and 180")
        if not (-90 <= south <= 90):
            errors.append("South coordinate must be between -90 and 90")
        if not (-180 <= east <= 180):
            errors.append("East coordinate must be between -180 and 180")
        if not (-90 <= north <= 90):
            errors.append("North coordinate must be between -90 and 90")
        
        if west >= east:
            errors.append("West coordinate must be less than east coordinate")
        if south >= north:
            errors.append("South coordinate must be less than north coordinate")
        
        return errors
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> List[str]:
        """Validate date range"""
        errors = []
        
        try:
            from datetime import datetime
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            if start >= end:
                errors.append("Start date must be before end date")
            
            # Check if dates are reasonable (not too old, not in future)
            now = datetime.now()
            if end > now:
                errors.append("End date cannot be in the future")
            
            # Sentinel-2 started in 2015
            if start.year < 2015:
                errors.append("Start date is before Sentinel-2 mission start (2015)")
            
        except ValueError as e:
            errors.append(f"Invalid date format: {e}")
        
        return errors


# Command-line interface for configuration
def main():
    """Command-line interface for Sentinel Hub configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentinel Hub Configuration Manager')
    parser.add_argument('--setup', action='store_true', help='Run interactive setup')
    parser.add_argument('--test', action='store_true', help='Test current credentials')
    parser.add_argument('--template', action='store_true', help='Create configuration template')
    parser.add_argument('--config-dir', type=str, help='Configuration directory')
    
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir) if args.config_dir else None
    config_manager = SentinelHubConfigManager(config_dir)
    
    if args.setup:
        success = config_manager.setup_interactive()
        exit(0 if success else 1)
    
    elif args.test:
        credentials = config_manager.load_credentials()
        if not credentials:
            print("❌ No credentials found. Run --setup first.")
            exit(1)
        
        print("🔍 Testing Sentinel Hub credentials...")
        result = config_manager.test_credentials(credentials)
        
        if result['valid']:
            print("✅ Credentials are valid!")
            print(f"Account ID: {result.get('account_id', 'N/A')}")
            print(f"Token expires in: {result.get('expires_in', 'N/A')} seconds")
        else:
            print(f"❌ Credentials test failed: {result['error']}")
            exit(1)
    
    elif args.template:
        template_path = config_manager.save_config_template()
        print(f"✅ Configuration template created: {template_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
