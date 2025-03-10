import os
import sys
import logging

# Define logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Create logs directory if not exists
log_dir = "logs"
log_filepath = os.path.join(log_dir, "runtime_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# Define the logger
logger = logging.getLogger("cnnClassifierLogger")
