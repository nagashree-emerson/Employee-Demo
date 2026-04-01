
# language: python

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("employee_assignment_agent_config")

# API base URL and endpoints
BASE_URL = "https://workforce.example.com"
ENDPOINTS = {
    "attendance_status": "/api/attendance/status",
    "employee_skills": "/api/employees/skills",
    "employee_capacity": "/api/employees/capacity",
    "task_priority": "/api/tasks/priority",
    "task_due_date": "/api/tasks/due-date",
    "task_dependencies": "/api/tasks/dependencies",
    "create_assignment": "/api/assignments",
    "utilization_summary": "/api/assignments/utilization-summary",
    "unassigned_tasks": "/api/assignments/unassigned-tasks"
}

# Default values and fallbacks
DEFAULT_LLM_CONFIG = {
    "provider": "azure",
    "model": "gpt-4.1",
    "temperature": 0.7,
    "max_tokens": 2000,
    "system_prompt": (
        "You are a professional Employee Work Assignment Agent. Assign daily work only to employees marked as available in attendance. "
        "Use task priority, due date, required skills, dependencies, and individual capacity to create balanced allocations. "
        "Exclude absent or leave employees and reduce capacity for half-day availability. Output assignments, a utilization summary, "
        "and clearly labeled unassigned tasks with reasons. Communicate decisions clearly and transparently."
    ),
    "user_prompt_template": "Please provide the list of tasks and employees for today's assignment. Ensure attendance, skills, and capacity data are up-to-date.",
    "few_shot_examples": [
        "Assign the following tasks: Task A (priority high, due today, skill X), Task B (priority medium, due tomorrow, skill Y). Employees: John (present, skill X), Jane (half-day, skill Y).",
        "Tasks: Task C (priority high, skill Z). Employees: Mike (absent, skill Z)."
    ]
}

DOMAIN_SETTINGS = {
    "domain": "general",
    "agent_name": "Employee Work Assignment Agent",
    "attendance_status_values": ["present", "half-day", "absent", "leave"],
    "capacity_adjustment_half_day": 0.5,
    "audit_log_retention_days": 365,
    "max_assignment_retries": 3,
    "assignment_error_codes": ["ERR_NO_AVAILABLE_EMPLOYEE", "ERR_INSUFFICIENT_CAPACITY"]
}

class ConfigError(Exception):
    pass

class AgentConfig:
    def __init__(self):
        # API keys and secrets
        self.WORKFORCE_API_TOKEN = os.getenv("WORKFORCE_API_TOKEN")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.PII_ENCRYPTION_KEY = os.getenv("PII_ENCRYPTION_KEY")
        # LLM config
        self.llm_config = DEFAULT_LLM_CONFIG.copy()
        # Domain settings
        self.domain_settings = DOMAIN_SETTINGS.copy()
        # API endpoints
        self.base_url = BASE_URL
        self.endpoints = ENDPOINTS.copy()
        # Validate on init
        self.validate()

    def validate(self):
        missing = []
        if not self.WORKFORCE_API_TOKEN:
            missing.append("WORKFORCE_API_TOKEN")
        if not self.AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.AZURE_OPENAI_DEPLOYMENT:
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if not self.PII_ENCRYPTION_KEY:
            missing.append("PII_ENCRYPTION_KEY")
        if missing:
            logger.error(f"Missing required API keys or secrets: {', '.join(missing)}")
            raise ConfigError(f"Missing required API keys or secrets: {', '.join(missing)}")

    def get_api_headers(self):
        if not self.WORKFORCE_API_TOKEN:
            raise ConfigError("WORKFORCE_API_TOKEN is missing")
        return {
            "Authorization": f"Bearer {self.WORKFORCE_API_TOKEN}",
            "Content-Type": "application/json"
        }

    def get_llm_config(self):
        return self.llm_config

    def get_domain_settings(self):
        return self.domain_settings

    def get_endpoint(self, key):
        return self.endpoints.get(key)

    def get_base_url(self):
        return self.base_url

    def get_pii_encryption_key(self):
        if not self.PII_ENCRYPTION_KEY:
            raise ConfigError("PII_ENCRYPTION_KEY is missing")
        return self.PII_ENCRYPTION_KEY

# Singleton config instance
try:
    agent_config = AgentConfig()
except ConfigError as e:
    logger.error(f"Agent configuration failed: {e}")
    # Optionally: raise or exit
    raise

# Example usage:
# headers = agent_config.get_api_headers()
# llm_cfg = agent_config.get_llm_config()
# endpoint = agent_config.get_endpoint("attendance_status")
# base_url = agent_config.get_base_url()
# encryption_key = agent_config.get_pii_encryption_key()
