import os
import logging
from pathlib import Path

def validate_file(file_path):
    """Validate if a file exists and has a supported extension"""
    return os.path.isfile(file_path) and file_path.lower().endswith(('.csv', '.xlsx', '.xls'))

def setup_logger(name='dataset_analyzer', log_level=logging.INFO):
    """Configure application-wide logger"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # File handler for logging to file
    file_handler = logging.FileHandler(logs_dir / "app.log")
    file_handler.setLevel(log_level)
    
    # Console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger():
    """Get or create application logger"""
    return logging.getLogger('dataset_analyzer')