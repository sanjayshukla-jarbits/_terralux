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
