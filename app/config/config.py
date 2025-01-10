
from dotenv import load_dotenv, find_dotenv
import os
import sys
import logging
from dataclasses import dataclass, field


cur_dir = os.getcwd()
parent_dir = os.path.realpath(os.path.join(os.path.dirname(cur_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    sys.path.append(cur_dir)
sys.path.insert(1, ".")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    return logger




logger = setup_logging()
@dataclass
class ConfigStaticFiles:
    """
    Configuration for static files and services.
    Managed through environment variables.
    """

    google_gemini_api_key:str =None
    mllm_name:str = None


    @classmethod
    def load_config(cls):
        """
        Load configuration from environment variables or `.env` files.
        """
        dotenv_path = find_dotenv(".env")
        if dotenv_path:
            load_dotenv(dotenv_path)
            logger.info(f"Loaded local .env file from {dotenv_path}")
        else:
            logger.warning("Local .env file not found.")

   
        dotenv_path_git = find_dotenv("app/env/.env")
        if dotenv_path_git and dotenv_path_git != dotenv_path:
            load_dotenv(dotenv_path_git)
            logger.info(f"Loaded Git .env file from {dotenv_path_git}")
        else:
            logger.warning("Git .env file not found or same as local.")


        cls.google_gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        cls.mllm_name= os.getenv("MLLM_MODEL_NAME")


        # Validate all required environment variables are set
        cls.validate_environment()

    def __post_init__(self):
        self.load_config()

    @classmethod
    def validate_environment(cls):
        """
        Validate that all required environment variables are set.
        """
        required_env_vars = [
            "google_gemini_api_key","mllm_name"]

        missing_vars = []
        for var in required_env_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
                logger.error(f"Environment variable {var} is not set.")

        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
        



ConfigStaticFiles.load_config()