"""
Data acquisition steps for _terralux orchestrator
"""

# Import existing mock implementations (keep for compatibility)
try:
    from .sentinel_hub_step import SentinelHubAcquisitionStep as MockSentinelHubStep
    from .dem_acquisition_step import DEMAcquisitionStep
    from .local_files_step import LocalFilesDiscoveryStep
except ImportError:
    MockSentinelHubStep = None
    DEMAcquisitionStep = None
    LocalFilesDiscoveryStep = None

# Import real implementation for _terralux
try:
    from .real_sentinel_hub_step import RealSentinelHubAcquisitionStep
    REAL_IMPLEMENTATION_AVAILABLE = True
except ImportError:
    RealSentinelHubAcquisitionStep = None
    REAL_IMPLEMENTATION_AVAILABLE = False

# Register steps with _terralux specific naming
try:
    from ..base.step_registry import StepRegistry
    
    # Register real implementation as primary
    if REAL_IMPLEMENTATION_AVAILABLE:
        StepRegistry.register('sentinel_hub_acquisition', RealSentinelHubAcquisitionStep)
        print("✓ Real Sentinel Hub step registered for _terralux")
    else:
        if MockSentinelHubStep:
            StepRegistry.register('sentinel_hub_acquisition', MockSentinelHubStep)
            print("⚠ Using mock Sentinel Hub step for _terralux")
    
    # Register other steps
    if DEMAcquisitionStep:
        StepRegistry.register('dem_acquisition', DEMAcquisitionStep)
    if LocalFilesDiscoveryStep:
        StepRegistry.register('local_files_discovery', LocalFilesDiscoveryStep)
        
except ImportError:
    print("⚠ Step registry not available")

__all__ = [
    'RealSentinelHubAcquisitionStep',
    'MockSentinelHubStep', 
    'DEMAcquisitionStep',
    'LocalFilesDiscoveryStep',
    'REAL_IMPLEMENTATION_AVAILABLE'
]
