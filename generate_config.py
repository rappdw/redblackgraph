#!/usr/bin/env python3
"""Generate __config__.py for redblackgraph package."""

import os
import sys

def generate_config():
    """Generate __config__.py with build configuration."""
    config_content = """# This file is generated during the build process
# It contains configuration information about how the package was built

def show():
    \"\"\"Show configuration information.\"\"\"
    print("RedBlackGraph configuration:")
    print("  Python version: {}")
    print("  Build system: Meson")
    
__all__ = ['show']
""".format(sys.version)
    
    config_path = os.path.join('redblackgraph', '__config__.py')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Generated {config_path}")

if __name__ == '__main__':
    generate_config()
