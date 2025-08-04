#!/usr/bin/env python3
"""
File Watcher Module for Geothermal Digital Twin AI

Monitors knowledge/corpus/ and data/ directories for new or updated files
based on modification time (mtime). Triggers appropriate ingest scripts
when changes are detected.

Usage:
    python src/file_watcher.py

Critical behavior:
- Runs at app start
- Compares current file mtimes against stored timestamp
- Triggers ingest_literature.py for corpus changes
- Triggers ingest_raw.py --all for data changes
- Updates timestamp stamp file after processing
"""

import sys
import time
import subprocess
from pathlib import Path
# typing imports not needed for basic types
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
STAMP_FILE = ".last_ingest"
KNOWLEDGE_CORPUS_DIR = Path("knowledge/corpus")
DATA_DIR = Path("data")


def setup_directories():
    """Ensure required directories exist."""
    KNOWLEDGE_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure verified")


def changed(dir_path: Path, stamp: float):
    """
    Check if any files in directory tree have been modified since timestamp.
    
    Args:
        dir_path: Directory to check recursively
        stamp: Timestamp to compare against (Unix timestamp)
        
    Returns:
        True if any file has mtime > stamp, False otherwise
    """
    if not dir_path.exists():
        logger.warning(f"Directory {dir_path} does not exist")
        return False
    
    try:
        # Check all files recursively
        changed_files = []
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                file_mtime = file_path.stat().st_mtime
                if file_mtime > stamp:
                    changed_files.append((file_path, file_mtime))
        
        if changed_files:
            logger.info(f"Found {len(changed_files)} changed files in {dir_path}")
            for file_path, mtime in changed_files[:5]:  # Log first 5 files
                logger.info(f"  Changed: {file_path} (mtime: {time.ctime(mtime)})")
            if len(changed_files) > 5:
                logger.info(f"  ... and {len(changed_files) - 5} more files")
            return True
        else:
            logger.info(f"No changes detected in {dir_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking changes in {dir_path}: {e}")
        return False


def run_ingest_script(script_path: str, args: list = None):
    """
    Run ingest script with proper error handling.
    
    Args:
        script_path: Path to Python script to run
        args: Additional arguments for the script
        
    Returns:
        True if script ran successfully, False otherwise
    """
    if args is None:
        args = []
    
    cmd = [sys.executable, script_path] + args
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully completed {script_path}")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"Script {script_path} failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Script {script_path} timed out after 5 minutes")
        return False
    except FileNotFoundError:
        logger.error(f"Script {script_path} not found")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running {script_path}: {e}")
        return False


def get_last_stamp():
    """
    Read last ingest timestamp from stamp file.
    
    Returns:
        Last timestamp as float, 0.0 if file doesn't exist
    """
    stamp_path = Path(STAMP_FILE)
    
    if stamp_path.exists():
        try:
            stamp_val = float(stamp_path.read_text().strip())
            logger.info(f"Last ingest timestamp: {time.ctime(stamp_val)}")
            return stamp_val
        except (ValueError, IOError) as e:
            logger.warning(f"Could not read stamp file: {e}. Using timestamp 0.0")
            return 0.0
    else:
        logger.info("No previous stamp file found. Will process all files.")
        return 0.0


def update_stamp():
    """Update stamp file with current timestamp."""
    current_time = time.time()
    stamp_path = Path(STAMP_FILE)
    
    try:
        stamp_path.write_text(str(current_time))
        logger.info(f"Updated stamp file with timestamp: {time.ctime(current_time)}")
    except IOError as e:
        logger.error(f"Could not update stamp file: {e}")


def main():
    """
    Main file watcher logic.
    
    1. Get last ingest timestamp
    2. Check for changes in corpus and data directories
    3. Trigger appropriate ingest scripts if changes detected
    4. Update timestamp stamp file
    """
    logger.info("=== Geothermal Digital Twin File Watcher Started ===")
    
    # Ensure directories exist
    setup_directories()
    
    # Get last ingest timestamp
    stamp_val = get_last_stamp()
    
    # Track if any ingestion was triggered
    ingestion_triggered = False
    
    # Check knowledge corpus for changes
    logger.info("Checking knowledge corpus for changes...")
    if changed(KNOWLEDGE_CORPUS_DIR, stamp_val):
        logger.info("Changes detected in knowledge corpus - triggering literature ingestion")
        success = run_ingest_script("src/ingest_literature.py")
        if success:
            ingestion_triggered = True
        else:
            logger.error("Literature ingestion failed")
    
    # Check data directory for changes
    logger.info("Checking data directory for changes...")
    if changed(DATA_DIR, stamp_val):
        logger.info("Changes detected in data directory - triggering raw data ingestion")
        success = run_ingest_script("src/ingest_raw.py", ["--all"])
        if success:
            ingestion_triggered = True
        else:
            logger.error("Raw data ingestion failed")
    
    # Update stamp file regardless of success/failure to prevent repeated attempts
    # on the same files
    update_stamp()
    
    if ingestion_triggered:
        logger.info("=== File Watcher Completed - Ingestion Triggered ===")
    else:
        logger.info("=== File Watcher Completed - No Changes Detected ===")


if __name__ == "__main__":
    main()