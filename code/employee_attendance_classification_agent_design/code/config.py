
import os
import logging
from typing import Optional, Dict, Any

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Configuration management for Employee Attendance Classification Agent.
    Handles environment variable loading, API key management, LLM config,
    domain settings, validation, error handling, and default values.
    """

    # --- Environment Variable Loading ---
    @staticmethod
    def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)

    # --- API Key Management ---
    @staticmethod
    def get_azure_openai_key() -> str:
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise ConfigError("Missing required Azure OpenAI API key (AZURE_OPENAI_API_KEY).")
        return key

    @staticmethod
    def get_azure_openai_endpoint() -> str:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ConfigError("Missing required Azure OpenAI endpoint (AZURE_OPENAI_ENDPOINT).")
        return endpoint

    @staticmethod
    def get_azure_openai_deployment() -> str:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise ConfigError("Missing required Azure OpenAI deployment name (AZURE_OPENAI_DEPLOYMENT).")
        return deployment

    # --- LLM Configuration ---
    LLM_CONFIG: Dict[str, Any] = {
        "provider": "azure",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are a professional attendance classification agent. Your task is to record and classify daily employee attendance using check-in logs, leave data, shift rules, and holiday calendars. Apply strict policy order and communicate results clearly, ensuring compliance and data privacy."
        ),
        "user_prompt_template": (
            "Please provide the employee check-in logs, leave data, shift rules, and holiday calendar for today's attendance classification."
        ),
        "few_shot_examples": [
            "Employee A checked in at 8:55 AM, shift starts at 9:00 AM, no leave, not a holiday. Employee A is classified as Present.",
            "Employee B checked in at 9:15 AM, shift starts at 9:00 AM, grace period is 10 minutes, no leave, not a holiday. Employee B is classified as Late Present.",
            "Employee C has approved leave for today. Employee C is classified as Leave."
        ]
    }

    # --- Domain-Specific Settings ---
    DOMAIN = "general"
    AGENT_NAME = "Employee Attendance Classification Agent"
    DB_URL = os.getenv("ATTENDANCE_AGENT_DB_URL", "sqlite:///./attendance_agent.db")
    LOG_LEVEL = os.getenv("ATTENDANCE_AGENT_LOG_LEVEL", "INFO")

    # --- Validation and Error Handling ---
    @classmethod
    def validate_llm_config(cls):
        errors = []
        try:
            cls.get_azure_openai_key()
        except ConfigError as e:
            errors.append(str(e))
        try:
            cls.get_azure_openai_endpoint()
        except ConfigError as e:
            errors.append(str(e))
        try:
            cls.get_azure_openai_deployment()
        except ConfigError as e:
            errors.append(str(e))
        if errors:
            raise ConfigError(" | ".join(errors))

    # --- Default Values and Fallbacks ---
    @staticmethod
    def get_grace_period() -> int:
        try:
            return int(os.getenv("ATTENDANCE_AGENT_GRACE_PERIOD", "10"))
        except Exception:
            return 10

    @staticmethod
    def get_half_day_cutoff() -> str:
        return os.getenv("ATTENDANCE_AGENT_HALF_DAY_CUTOFF", "12:00")

    @staticmethod
    def get_audit_log_retention_days() -> int:
        try:
            return int(os.getenv("ATTENDANCE_AGENT_AUDIT_RETENTION_DAYS", "2555"))
        except Exception:
            return 2555

    # --- Logging Setup ---
    @classmethod
    def setup_logging(cls):
        level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        )

# Example usage:
# try:
#     Config.validate_llm_config()
# except ConfigError as e:
#     print(f"Configuration error: {e}")
#     exit(1)
# Config.setup_logging()
