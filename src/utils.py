import time
from functools import wraps
from src.logger import get_logger

logger = get_logger("Timer")

def time_research_task(func):
    """Decorator that reports the execution time."""
    @wraps(func) # This preserves the metadata (name, docs) of the original function
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        result = func(*args, **kwargs)  # Run the actual function (e.g., data_loader)
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Execution of {func.__name__} completed in {duration:.2f} seconds")
        return result
    return wrapper
