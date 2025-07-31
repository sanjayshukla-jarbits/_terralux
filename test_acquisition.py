# Test the fixed constructor
step_config = {
    'type': 'sentinel_hub_acquisition',
    'hyperparameters': {
        'bbox': [85.30, 27.60, 85.32, 27.62],
        'use_real_api': True,
        'fallback_to_mock': True
    }
}
step = RealSentinelHubAcquisitionStep('test_step', step_config)
result = step.execute()
