import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Settings object that loads secrets and personalized configurations from .env file.
    
    This class automatically loads environment variables from a .env file in the project root
    and provides typed access to all configuration values used throughout the project.
    """
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_api_key: Optional[str] = Field(
        default=None,
        description="Weights & Biases API key for authentication"
    )
    wandb_entity: Optional[str] = Field(
        default=None,
        description="Weights & Biases entity/organization name for logging experiments"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for autointerp and other AI-powered features"
    )
    together_ai_api_key: Optional[str] = Field(
        default=None,
        description="TogetherAI API key for autointerp and other AI-powered features"
    )
    hf_access_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for accessing models and datasets"
    )

    # Path configurations
    output_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "output",
        description="Directory for caching SAE models and checkpoints"
    )

    def __init__(self, env_file: Optional[str | Path] = None, **kwargs):
        """Initialize settings by loading from .env file and environment variables.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in project root.
            **kwargs: Additional keyword arguments to override settings.
        """
        # Load environment variables from .env file
        if env_file is None:
            # Look for .env in the project root (same directory as this settings.py file)
            project_root = Path(__file__).parent
            env_file = project_root / ".env"
        
        # Load .env file if it exists, otherwise create it from template and raise error
        if Path(env_file).exists():
            load_dotenv(env_file, override=True)
        else:
            # Create default .env file from template
            with open(env_file, "w") as f:
                f.write(self.get_env_template())
            
            raise FileNotFoundError(
                f"Configuration file not found at {env_file}. "
                "A template has been created. Please fill in your API keys and configuration."
            )
        
        # Extract values from environment variables
        env_values = {
            "wandb_api_key": os.getenv("WANDB_API_KEY"),
            "wandb_entity": os.getenv("WANDB_ENTITY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "together_ai_api_key": os.getenv("TOGETHER_AI_API_KEY"),
            "hf_access_token": os.getenv("HF_ACCESS_TOKEN") or os.getenv("HF_TOKEN"),
            "output_dir": os.getenv("OUTPUT_DIR"),
        }
        
        # Remove None values and update with any provided kwargs
        env_values = {k: v for k, v in env_values.items() if v is not None}
        env_values.update(kwargs)
        
        super().__init__(**env_values)
    
    @property
    def repo_root(self) -> Path:
        """Get the repository root directory."""
        return Path(__file__).parent
    
    def is_ci(self) -> bool:
        """Check if running in a CI environment."""
        return os.getenv("CI") is not None
    
    def has_wandb_config(self) -> bool:
        """Check if Weights & Biases is properly configured."""
        return self.wandb_api_key is not None and self.wandb_entity is not None
    
    def has_openai_config(self) -> bool:
        """Check if OpenAI API is properly configured."""
        return self.openai_api_key is not None
    
    def has_hf_config(self) -> bool:
        """Check if Hugging Face token is properly configured."""
        return self.hf_access_token is not None
    
    @staticmethod
    def get_env_template() -> str:
        """Generate a template .env file with all available settings.
        
        Returns:
            String content for a .env template file with comments.
        """
        return """# SAE-Interpretability Configuration
# Copy this file to .env and fill in your values

# Weights & Biases Configuration
# Get your entity name from https://wandb.ai/settings
WANDB_ENTITY=your-wandb-entity

# API Keys
# Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-api-key

# TogetherAI API key for autointerp and other AI-powered features
# Get from https://api.together.ai/settings/api-keys
TOGETHER_AI_API_KEY=your-together-ai-api-key

# Hugging Face access token for accessing models/datasets
# Get from https://huggingface.co/settings/tokens
HF_ACCESS_TOKEN=your-hf-token

# Path Configuration
# Directory for caching SAE models (defaults to ./output)
OUTPUT_DIR=./output
"""


# Create a global settings instance
settings = Settings()
