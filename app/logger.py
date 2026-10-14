import os 
import logging
from app.config import settings 

def configure_logging() -> None:
    log_dir = os.path.join(settings.logging_dir)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, "app.log"),
                encoding="utf-8",                
            ),
            logging.StreamHandler()
        ]
    )

    
