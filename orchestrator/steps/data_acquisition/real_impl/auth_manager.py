"""
Sentinel Hub Authentication Manager
"""
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

class SentinelHubAuth:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires_at = None
        self.logger = logging.getLogger("SentinelHub.Auth")
    
    def get_token(self) -> str:
        """Get valid access token"""
        if self._is_token_valid():
            return self.access_token
        return self._refresh_token()
    
    def _is_token_valid(self) -> bool:
        """Check if current token is valid"""
        if not self.access_token or not self.token_expires_at:
            return False
        return datetime.now() < (self.token_expires_at - timedelta(minutes=5))
    
    def _refresh_token(self) -> str:
        """Refresh access token"""
        auth_url = "https://services.sentinel-hub.com/auth/oauth/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        response = requests.post(auth_url, data=data, timeout=30)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data['access_token']
        expires_in = token_data.get('expires_in', 3600)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        self.logger.info("Access token refreshed")
        return self.access_token
