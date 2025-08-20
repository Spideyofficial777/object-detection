import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

logger = logging.getLogger(__name__)

def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Remove files older than max_age_hours from directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours before file deletion
        
    Returns:
        Number of files deleted
    """
    if not os.path.exists(directory):
        return 0
    
    cutoff_time = time.time() - (max_age_hours * 3600)
    deleted_count = 0
    
    try:
        for file_path in Path(directory).iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
                    
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleanup: removed {deleted_count} files from {directory}")
    
    return deleted_count

def scheduled_cleanup():
    """Run scheduled cleanup of upload and results folders"""
    from config import Config
    
    logger.info("Running scheduled file cleanup...")
    
    upload_deleted = cleanup_old_files(Config.UPLOAD_FOLDER, Config.CLEANUP_AGE_HOURS)
    results_deleted = cleanup_old_files(Config.RESULTS_FOLDER, Config.CLEANUP_AGE_HOURS)
    
    total_deleted = upload_deleted + results_deleted
    if total_deleted > 0:
        logger.info(f"Scheduled cleanup completed: {total_deleted} files removed")

def schedule_cleanup(app):
    """Schedule periodic cleanup using APScheduler"""
    from config import Config
    
    scheduler = BackgroundScheduler()
    
    # Schedule cleanup every interval
    scheduler.add_job(
        func=scheduled_cleanup,
        trigger="interval",
        hours=Config.CLEANUP_INTERVAL_HOURS,
        id='file_cleanup'
    )
    
    scheduler.start()
    logger.info(f"Scheduled cleanup every {Config.CLEANUP_INTERVAL_HOURS} hours")
    
    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())