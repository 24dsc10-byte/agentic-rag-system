"""
Configuration management for Agentic RAG System
"""

import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env file
    )
    
    # Groq Configuration
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    
    # Application Configuration
    app_name: str = "Agentic RAG System"
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Vector Database
    vector_db_type: str = os.getenv("VECTOR_DB_TYPE", "in_memory")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    max_results: int = int(os.getenv("MAX_RESULTS", "5"))
    
    # Memory Configuration
    enable_memory: bool = os.getenv("ENABLE_MEMORY", "True").lower() == "true"
    memory_type: str = os.getenv("MEMORY_TYPE", "in_memory")
    history_retention_days: int = int(os.getenv("HISTORY_RETENTION_DAYS", "30"))
    
    # Server Configuration
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "4"))
    
    # Logging
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")


# Global settings instance
settings = Settings()
