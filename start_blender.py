import sys

# Configuration: Update this path to point to your virtual environment's site-packages
# Examples:
#   - Linux/Mac: '/path/to/venv/lib/python3.11/site-packages'
#   - Windows: 'C:\\path\\to\\venv\\Lib\\site-packages'

VIRTUAL_ENV_SITE_PACKAGES = '/path/to/your/virtualenv/lib/python3.11/site-packages'

# Add your virtualenv's site-packages path:
sys.path.append(VIRTUAL_ENV_SITE_PACKAGES)

import numpy as np
print("NumPy version inside Blender:", np.__version__)

# Verify Blender integration
import bpy
print("Blender is running this script successfully.")

# To run the optimised script, uncomment the line below:
# exec(open("gaussian_blender_script.py").read())