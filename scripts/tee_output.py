"""
Simple tee-like utility that writes both to console and log file.
Usage: python tee_output.py <log_file> <script> [args...]
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tee_output.py <log_file> <script> [args...]")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    script_args = sys.argv[2:]
    
    # Open log file
    with open(log_file, 'w', encoding='utf-8', buffering=1) as log:
        # Run the script
        process = subprocess.Popen(
            script_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read output line by line
        for line in process.stdout:
            # Write to console
            print(line, end='', flush=True)
            # Write to log file
            log.write(line)
            log.flush()
        
        # Wait for process to complete
        process.wait()
        sys.exit(process.returncode)
