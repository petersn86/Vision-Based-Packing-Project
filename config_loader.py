##############################################
# @Author: Peter Nolan
# @Contributor(s):
# @Document: 'config_loader.py'
#
# Description:
# Utility module for loading and accessing
# project configuration from config.yaml
#
# FIXED: Added encoding='utf-8' to config file open
#        to prevent Windows charmap errors.
#
##############################################

import yaml
import os
from pathlib import Path
from typing import Any, Dict

class Config:
    """Configuration manager for the Vision-Based Packing Project"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config.yaml file
        """
        from pathlib import Path
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Try parent directory (for when running from src/)
            parent_config = Path('..') / config_path
            if parent_config.exists():
                config_file = parent_config
            elif Path.cwd() / config_path != config_file:
                config_file = Path.cwd() / config_path
        
        self.config_path = str(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a config.yaml file in the project root."
            )
        
        # ---- FIX: encoding='utf-8' prevents Windows charmap error ----
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['video', 'paths', 'detection', 'llama']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'video.frame_interval')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_path(self, key_path: str, create: bool = False) -> str:
        """
        Get a path from config and optionally create the directory
        
        Args:
            key_path: Dot-separated path to config value
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Absolute path string
        """
        path = self.get(key_path)
        if path is None:
            raise ValueError(f"Path not found in config: {key_path}")
        
        if create:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        return path
    
    # Convenience properties for commonly used values
    @property
    def frame_interval(self) -> float:
        return self.get('video.frame_interval', 2.0)
    
    @property
    def frames_dir(self) -> str:
        return self.get('paths.frames_dir', 'data/frames')
    
    @property
    def yolo_model(self) -> str:
        return self.get('detection.yolo_model', 'yolo11m.pt')
    
    @property
    def confidence_threshold(self) -> float:
        return self.get('detection.confidence_threshold', 0.35)
    
    @property
    def llama_model(self) -> str:
        return self.get('llama.model_name', 'llama3.2-vision')
    
    @property
    def llama_enabled(self) -> bool:
        return self.get('llama.enabled', True)
    
    @property
    def tracking_enabled(self) -> bool:
        return self.get('tracking.enabled', True)
    
    @property
    def qr_enabled(self) -> bool:
        return self.get('qr_detection.enabled', True)
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global config instance (initialized when imported)
_config_instance = None

def load_config(config_path: str = "config.yaml") -> Config:
    """Load or reload the global configuration"""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance