import os
import sys

# Global reference to the active log file
LOG_FILE_GLOBAL = None


# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------

def setup_logging(results_dir, log_file):
    """
    Sets up the logging system.
    - Ensures results directory exists.
    - Creates a fresh log file.
    - Stores the log file path in a global variable for log_print().
    """
    global LOG_FILE_GLOBAL
    LOG_FILE_GLOBAL = log_file

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Also ensure plots subfolder exists
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Create/reset log file
    with open(log_file, "w") as f:
        f.write("=== Analysis Output Log ===\n\n")


def log_print(message):
    """
    Print to console AND write to the log file.
    """
    print(message)
    if LOG_FILE_GLOBAL is not None:
        with open(LOG_FILE_GLOBAL, "a") as f:
            f.write(message + "\n")


# ------------------------------------------------------------------------------
# Tee class — duplicates stdout to file (captures all print() calls)
# ------------------------------------------------------------------------------

class Tee:
    """
    Duplicate all print() output to multiple output streams.
    Used in main.py to send all terminal output to the log file.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()
