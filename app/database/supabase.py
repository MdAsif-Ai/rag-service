import functools
from typing import Any
from supabase import create_client, Client
from loguru import logger

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationException


@functools.lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Initializes and returns a cached Supabase client.
    Uses the service role key for backend administrative tasks.
    The key is never logged or exposed to the API responses.
    """
    settings: Settings = get_settings()
    
    try:
        # Pydantic AnyHttpUrl needs to be cast to string for the Supabase SDK
        url = str(settings.SUPABASE_URL)
        
        # Safely extract the secret value without logging it
        key = settings.SUPABASE_SERVICE_ROLE_KEY.get_secret_value()
        
        if not url or not key:
            raise ConfigurationException("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing in configuration.")
            
        client: Client = create_client(url, key)
        
        logger.info("Supabase client initialized successfully.")
        return client
        
    except ConfigurationException:
        raise
    except Exception as e:
        # Catch potential SDK initialization errors
        logger.error("Failed to initialize Supabase client.")
        raise ConfigurationException("Supabase client initialization failed.", detail=str(e))