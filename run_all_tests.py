#!/usr/bin/env python3
"""
Run all tests for the trading bot.
"""

import unittest
import subprocess
import sys
import glob
import os


def run_all_tests(verbose=False):
    """Run all tests by manually specifying test modules.
    
    Args:
        verbose: Whether to show detailed output
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Get all test files
    test_files = glob.glob('tests/**/test_*.py', recursive=True)
    
    # Sort files to ensure consistent order
    test_files.sort()
    
    # Build command
    cmd = [sys.executable, '-m', 'unittest']
    cmd.extend(test_files)
    
    # Run tests
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all trading bot tests.')
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Show detailed test output'
    )
    
    args = parser.parse_args()
    
    # Run tests
    success = run_all_tests(args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
