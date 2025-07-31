# Quick Start Guide: Real Sentinel Hub Implementation

## 🚀 Complete Setup in 5 Minutes

This guide will get you from mock data to real Sentinel Hub API integration in just a few steps.

## Step 1: Install Dependencies

```bash
# Navigate to your project directory
cd /home/ubuntu/_terralux

# Run the automated setup script
chmod +x setup_sentinel_hub.sh
./setup_sentinel_hub.sh
```

## Step 2: Get Sentinel Hub Credentials

1. Go to [Sentinel Hub Dashboard](https://apps.sentinel-hub.com/)
2. Create account or sign in
3. Create a new configuration 
4. Copy your **Client ID** and **Client Secret**

## Step 3: Set Environment Variables

```bash
# Set your credentials
export SENTINEL_HUB_CLIENT_ID="your-client-id-here"
export SENTINEL_HUB_CLIENT_SECRET="your-client-secret-here"

# Optional: Add to your shell profile for persistence
echo 'export SENTINEL_HUB_CLIENT_ID="your-client-id-here"' >> ~/.bashrc
echo 'export SENTINEL_HUB_CLIENT_SECRET="your-client-secret-here"' >> ~/.bashrc
source ~/.bashrc
```

## Step 4: Test the Setup

```bash
# Test basic setup
python test_sentinel_hub_real.py

# Test real data acquisition (small area, fast download)
python test_real_acquisition.py

# Run comprehensive demo
python integration_example.py
```

## Step 5: Use in Your Pipeline

### Option A: JSON Process Definition

Create `my_real_process.json`:
```json
{
  "process_info": {
    "name": "My Real Data Pipeline",
    "version": "1.0.0"
  },
  "steps": [
    {
      "id": "get_satellite_data",
      "type": "sentinel_hub_acquisition",
      "hyperparameters": {
        "bbox": [85.30, 27.60, 85.32, 27.62],
        "start_date": "2023-06-01",
        "end_date": "2023-06-07",
        "data_collection": "SENTINEL-2-L2A",
        "resolution": 10,
        "bands": ["B02", "B03", "B04", "B08"],
        "max_cloud_coverage": 20,
        "use_cache": true
      }
    }
  ]
}
```

Execute with:
```bash
python integration_example.py
```

### Option B: Direct Python Usage

```python
from orchestrator.steps.data_acquisition.real_sentinel_hub_step import RealSentinelHubAcquisitionStep

# Configure step
config = {
    'id': 'my_data',
    'type': 'sentinel_hub_acquisition', 
    'hyperparameters': {
        'bbox': [85.30, 27.60, 85.32, 27.62],
        'start_date': '2023-06-01',
        'end_date': '2023-06-07',
        'data_collection': 'SENTINEL-2-L2A',
        'resolution': 10,
        'bands': ['B02', 'B03', 'B04', 'B08']
    }
}

# Create and execute step
step = RealSentinelHubAcquisitionStep(
    config['id'], 
    config['type'], 
    config['hyperparameters']
)

result = step.execute()
print(f"Downloaded: {result['imagery_data']}")
```

## 🎯 What You Get

### ✅ Real API Integration
- Actual Sentinel Hub API calls
- OAuth2 authentication handling
- Automatic retry and error handling

### ✅ Smart Caching
- Automatic file caching
- Avoid redundant downloads
- Configurable cache location and size

### ✅ Graceful Fallbacks
- Falls back to mock data if API unavailable
- Continues processing even if some steps fail
- Clear error reporting and logging

### ✅ Production Ready
- Comprehensive error handling
- Performance monitoring
- Security best practices

## 📊 Example Output

```
🛰️ ENHANCED SENTINEL HUB INTEGRATION DEMO
==================================================
📡 Real API credentials: ✓ Available
🔄 Will use: Real API with mock fallback

📍 Test Parameters:
   Area: nepal_enhanced_test
   Bbox: [85.3, 27.6, 85.32, 27.62]
   Dates: 2023-06-01 to 2023-06-03

🚀 Executing enhanced process...

📊 EXECUTION RESULTS:
==============================
Process: Enhanced Data Acquisition with Real Sentinel Hub
Status: success
Execution time: 45.23s
Steps executed: 3

📈 Performance Breakdown:
   Real API calls: 1
   Mock fallbacks: 2
   Cache hits: 0

📋 Step Results:
   ✓ acquire_sentinel_real: completed (real)
      File size: 12.45 MB
      Data shape: (256, 256, 4)
   ✓ acquire_dem_srtm: completed (mock)
   ✓ discover_local_data: completed (mock)

📁 Generated Artifacts:
   - acquire_sentinel_real: imagery_data (real)
     Path: /home/ubuntu/.sentinel_hub_cache/data/a1b2c3d4.tif
   - acquire_dem_srtm: elevation_data (mock)
   - discover_local_data: discovered_files (mock)

✅ All steps completed successfully!
🎉 Successfully used real Sentinel Hub API!
```

## 🛠️ Configuration Options

### Data Collections
```python
# Sentinel-2 (optical)
"data_collection": "SENTINEL-2-L2A"  # Atmospherically corrected
"data_collection": "SENTINEL-2-L1C"  # Top-of-atmosphere

# Sentinel-1 (SAR)
"data_collection": "SENTINEL-1-GRD"  # Ground Range Detected
```

### Bands
```python
# Sentinel-2 common bands
"bands": ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
"bands": ["B05", "B06", "B07", "B8A", "B11", "B12"]  # Additional bands

# Sentinel-1 polarizations
"bands": ["VV", "VH"]  # Dual polarization
```

### Resolution
```python
"resolution": 10   # 10m (Sentinel-2: B02,B03,B04,B08)
"resolution": 20   # 20m (Sentinel-2: B05,B06,B07,B8A,B11,B12)  
"resolution": 60   # 60m (faster downloads)
```

## 🔧 Troubleshooting

### Authentication Issues
```bash
# Check credentials
echo $SENTINEL_HUB_CLIENT_ID
echo $SENTINEL_HUB_CLIENT_SECRET

# Test authentication
python -c "
from orchestrator.steps.data_acquisition.sentinel_hub_config import SentinelHubConfigManager
config = SentinelHubConfigManager()
result = config.test_credentials()
print('Valid:', result['valid'])
if not result['valid']:
    print('Error:', result['error'])
"
```

### No Data Found
```bash
# Check if data exists for your area and time period
# Reduce cloud coverage threshold
"max_cloud_coverage": 80  # Allow more clouds

# Extend time range
"start_date": "2023-01-01"
"end_date": "2023-12-31"
```

### Quota Issues
- Check usage at [Sentinel Hub Dashboard](https://apps.sentinel-hub.com/)
- Reduce resolution: `"resolution": 60`
- Use smaller areas: smaller bbox
- Enable caching: `"use_cache": true`

## 📈 Performance Tips

1. **Start Small**: Test with small areas and short time ranges
2. **Use Caching**: Enable `"use_cache": true` to avoid re-downloads
3. **Appropriate Resolution**: Higher resolution = larger files and longer downloads
4. **Cloud Filtering**: Lower cloud coverage = better quality but fewer images
5. **Monitor Quota**: Check processing unit usage regularly

## 🎉 Success! You're Ready

You now have:
- ✅ Real Sentinel Hub API integration
- ✅ Smart caching and fallbacks  
- ✅ Production-ready error handling
- ✅ JSON process definitions
- ✅ Performance monitoring

**Next**: Extend to preprocessing, feature extraction, and modeling steps!

## 📚 Additional Resources

- [Sentinel Hub Documentation](https://docs.sentinel-hub.com/)
- [API Reference](https://docs.sentinel-hub.com/api/latest/)
- [Community Forum](https://forum.sentinel-hub.com/)
- [Processing Examples](https://docs.sentinel-hub.com/api/latest/evalscript/)

---

🚀 **Ready to process real satellite data at scale!**
