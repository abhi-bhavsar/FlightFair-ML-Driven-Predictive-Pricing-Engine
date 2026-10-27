import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from src.config import LOGS_DIR

LOG_FILE = f"flightfair_{datetime.now().strftime('%Y_%m_%d')}.log"
LOG_FILE_PATH = LOGS_DIR / LOG_FILE

# Reconfigure stdout to utf-8 if possible on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    format="[ %(asctime)s ] [ %(levelname)s ] [ %(filename)s:%(lineno)d - %(funcName)s() ] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FlightFair")
