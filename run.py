#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ME-Benchmark: Model Editing Benchmark Framework
Main entry point
"""

import argparse
import os
import sys

from me_benchmark.utils.config import load_config
from me_benchmark.factory import create_runner


def main():
    parser = argparse.ArgumentParser(description='ME-Benchmark: Model Editing Benchmark Framework')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--work-dir', type=str, default='.', help='Work directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize runner using factory
    runner_config = config.get('runner', {'type': 'local'})
    runner = create_runner(runner_config, config)
    
    # Run evaluation
    runner.run()


if __name__ == '__main__':
    main()