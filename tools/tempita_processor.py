#!/usr/bin/env python3
"""
Tempita template processor for RedBlackGraph build system.
Wrapper around tempita to match NumPy's usage pattern.
"""
import argparse
import sys
from pathlib import Path


def process_template(input_file, output_file):
    """Process a .c.in or .h.in Tempita template."""
    try:
        import tempita
    except ImportError:
        print("Error: tempita not installed. Run: pip install tempita", 
              file=sys.stderr)
        return 1
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Read and process template
    with open(input_path, 'r', encoding='utf-8') as f:
        template_str = f.read()
    
    template = tempita.Template(template_str, name=str(input_path))
    result = template.substitute()
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"Generated {output_path} from {input_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Process Tempita templates for RedBlackGraph'
    )
    parser.add_argument('input', help='Input template file (.c.in or .h.in)')
    parser.add_argument('-o', '--outfile', required=True, help='Output file')
    
    args = parser.parse_args()
    return process_template(args.input, args.outfile)


if __name__ == '__main__':
    sys.exit(main())
