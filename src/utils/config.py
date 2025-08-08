"""Configuration loader for Fed Minutes project."""

import yaml
import os
from pathlib import Path

def load_config(config_path=None):
    """Load configuration from config.yaml."""
    if config_path is None:
        # Find project root (where config/ directory is)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute paths based on project root
    project_root = config_path.parent.parent
    for key in ['pdf_dir', 'txt_dir', 'processed_dir', 'vector_db']:
        if key in config['paths']:
            config['paths'][key] = str(project_root / config['paths'][key])
    
    return config