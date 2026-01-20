"""
Settings module for configuration and API keys.

This module provides configuration settings that can be set via environment variables.
Create a .env file or set environment variables to configure these settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # Wandb configuration
    wandb_api_key: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # HuggingFace configuration
    hf_access_token: Optional[str] = None
    
    # OpenAI configuration
    openai_api_key: Optional[str] = None
    
    # Together AI configuration
    together_ai_api_key: Optional[str] = None
    
    # Output directory
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Load settings from environment variables."""
        # Wandb
        self.wandb_api_key = os.environ.get("WANDB_API_KEY", self.wandb_api_key)
        self.wandb_entity = os.environ.get("WANDB_ENTITY", self.wandb_entity)
        
        # HuggingFace
        self.hf_access_token = os.environ.get("HF_ACCESS_TOKEN", self.hf_access_token)
        if self.hf_access_token is None:
            self.hf_access_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        
        # OpenAI
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", self.openai_api_key)
        
        # Together AI
        self.together_ai_api_key = os.environ.get("TOGETHER_API_KEY", self.together_ai_api_key)
        
        # Output directory
        output_dir_str = os.environ.get("OUTPUT_DIR", None)
        if output_dir_str:
            self.output_dir = Path(output_dir_str)
        elif self.output_dir is None:
            # Default output directory
            self.output_dir = Path("./outputs")
    
    def has_hf_config(self) -> bool:
        """Check if HuggingFace credentials are configured."""
        return self.hf_access_token is not None


# Create global settings instance
settings = Settings()
