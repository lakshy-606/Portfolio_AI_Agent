"""Project-wide paths and settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCHEMA_PATH = PROJECT_ROOT / "database_schema.sql"
GROUND_TRUTH_PATH = PROJECT_ROOT / "ground_truth_dataset.json"

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Upper bound on rows returned to the LLM from a single SQL query.
MAX_SQL_ROWS = 200

# Graph steps per question (each model call and each tool round counts as one).
# Leaves room for a few SQL retries without letting the agent loop indefinitely.
MAX_AGENT_STEPS = 12
