#!/usr/bin/env python3
"""
Install Optional Dependencies Script
===================================
Installs optional dependencies based on user requirements.
"""

import subprocess
import sys
from typing import Dict, List

OPTIONAL_DEPENDENCIES = {
    'modeling': [
        'kneed',  # For K-means clustering optimization
        'optuna',  # For hyperparameter tuning
        'xgboost',  # For advanced modeling
    ],
    'prediction': [
        'scipy',  # Full scipy including interpolation
        'scikit-learn>=1.0',  # Latest sklearn features
    ],
    'reporting': [
        'weasyprint',  # PDF generation
        'reportlab',  # Alternative PDF generation
        'jinja2',  # Template rendering
    ],
    'geospatial': [
        'sentinelhub',  # Sentinel Hub API
        'elevation',  # DEM data access
        'earthpy',  # Earth science utilities
    ],
    'visualization': [
        'plotly>=5.0',  # Interactive plots
        'bokeh',  # Interactive visualization
        'seaborn',  # Statistical plotting
    ]
}

def install_packages(packages: List[str]) -> bool:
    """Install a list of packages using pip."""
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    return True

def main():
    """Main installation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Install optional TerraLux dependencies')
    parser.add_argument('--category', choices=list(OPTIONAL_DEPENDENCIES.keys()) + ['all'],
                       help='Category of dependencies to install')
    parser.add_argument('--list', action='store_true', help='List available categories')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available dependency categories:")
        for category, packages in OPTIONAL_DEPENDENCIES.items():
            print(f"  {category}: {', '.join(packages)}")
        return
    
    if not args.category:
        parser.print_help()
        return
    
    if args.category == 'all':
        all_packages = []
        for packages in OPTIONAL_DEPENDENCIES.values():
            all_packages.extend(packages)
        success = install_packages(all_packages)
    else:
        packages = OPTIONAL_DEPENDENCIES.get(args.category, [])
        if not packages:
            print(f"Unknown category: {args.category}")
            return
        success = install_packages(packages)
    
    if success:
        print(f"✅ All {args.category} dependencies installed successfully!")
    else:
        print(f"❌ Some dependencies failed to install.")

if __name__ == '__main__':
    main()
