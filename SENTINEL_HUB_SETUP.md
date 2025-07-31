# Sentinel Hub Real Data Acquisition Setup

## Overview

This setup provides real Sentinel Hub API integration for satellite data acquisition, replacing mock implementations with actual API calls.

## Features

- ✅ Real Sentinel Hub API integration
- ✅ OAuth2 authentication handling
- ✅ Automatic data download and caching
- ✅ Support for Sentinel-1 and Sentinel-2
- ✅ Configurable resolution and bands
- ✅ Cloud filtering and quality control
- ✅ Graceful fallback to mock data
- ✅ Progress tracking and error handling

## Quick Start

### 1. Get Sentinel Hub Credentials

1. Go to [Sentinel Hub Dashboard](https://apps.sentinel-hub.com/)
2. Create an account or log in
3. Create a new configuration
4. Note your Client ID and Client Secret

### 2. Set Environment Variables

```bash
export SENTINEL_HUB_CLIENT_ID="your-client-id-here"
export SENTINEL_HUB_CLIENT_SECRET="your-client-secret-here"
```

### 3. Test the Setup

```bash
# Test basic setup
python test_sentinel_hub_real.py

# Test real data acquisition
python test_real_acquisition.py

# Validate complete setup
python validate_setup.py
```

## Usage Examples

### Basic Real Data Acquisition

```python
from orchestrator.steps.data_acquisition.real_sentinel_hub_step import RealSentinelHubAcquisitionStep

step_config = {
    'id': 'acquire_sentinel',
    'type': 'sentinel_hub_acquisition',
    'hyperparameters': {
        'bbox': [85.30, 27.60, 85.32, 27.62],
        'start_date': '2023-06-01',
        'end_date': '2023-06-07',
        'data_collection': 'SENTINEL-2-L2A',
        'resolution': 10,
        'bands': ['B02', 'B03', 'B04', 'B08'],
        'max_cloud_coverage': 20,
        'client_id': 'your-client-id',
        'client_secret': 'your-client-secret'
    }
}

step = RealSentinelHubAcquisitionStep(
    step_config['id'],
    step_config['type'],
    step_config['hyperparameters']
)

result = step.execute()
```

### JSON Process Definition

```json
{
  "process_info": {
    "name": "Real Sentinel Data Acquisition",
    "version": "1.0.0"
  },
  "steps": [
    {
      "id": "acquire_sentinel",
      "type": "sentinel_hub_acquisition",
      "hyperparameters": {
        "bbox": "{bbox}",
        "start_date": "{start_date}",
        "end_date": "{end_date}",
        "data_collection": "SENTINEL-2-L2A",
        "resolution": 10,
        "bands": ["B02", "B03", "B04", "B08"],
        "max_cloud_coverage": 20,
        "use_cache": true,
        "fallback_to_mock": false
      }
    }
  ]
}
```

## Configuration Options

### Data Collections
- `SENTINEL-2-L2A`: Atmospherically corrected Sentinel-2 data
- `SENTINEL-2-L1C`: Top-of-atmosphere Sentinel-2 data
- `SENTINEL-1-GRD`: Ground Range Detected SAR data

### Bands
- **Sentinel-2**: B01-B12, B8A
- **Sentinel-1**: VV, VH, HH, HV

### Resolution Options
- Sentinel-2: 10m, 20m, 60m
- Sentinel-1: 10m, 40m

## Caching

Data is automatically cached to avoid redundant downloads:
- Cache location: `~/.sentinel_hub_cache/`
- Cache key: MD5 hash of request parameters
- Automatic cleanup of old data

## Error Handling

The system includes comprehensive error handling:
- Authentication failures
- Network timeouts
- Invalid requests
- Quota limits
- Automatic fallback to mock data (configurable)

## Troubleshooting

### Common Issues

1. **Authentication Error**
   - Check your Client ID and Client Secret
   - Verify account status on Sentinel Hub Dashboard

2. **No Data Found**
   - Check date range (ensure data exists for the period)
   - Adjust cloud coverage threshold
   - Verify bounding box coordinates

3. **Quota Exceeded**
   - Check processing unit usage on dashboard
   - Consider using cached data
   - Reduce resolution or area size

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Use appropriate resolution**: Higher resolution = larger files
2. **Limit time range**: Shorter periods = faster processing
3. **Use cloud filtering**: Lower cloud coverage = better quality
4. **Enable caching**: Avoid redundant downloads
5. **Batch requests**: Process multiple areas together

## Security

- Credentials are encrypted when stored locally
- Support for system keyring integration
- Environment variables for CI/CD
- No hardcoded credentials in code

## Support

For issues related to:
- **Sentinel Hub API**: [Sentinel Hub Support](https://forum.sentinel-hub.com/)
- **This Implementation**: Check logs and error messages
- **Quota Questions**: Sentinel Hub Dashboard

