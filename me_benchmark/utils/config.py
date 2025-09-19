"""
Configuration utilities for ME-Benchmark
"""
import yaml
import json
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to a file"""
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.endswith('.json'):
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")