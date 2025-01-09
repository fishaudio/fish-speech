import sys
import subprocess
import os

#!/usr/bin/env python

def main():
    # Make path relative to this file
    script_path = os.path.join(
        os.path.dirname(__file__),
        "../../fish_speech/models/vqgan/inference.py"
    )
    subprocess.run(["python", script_path] + sys.argv[1:])

if __name__ == "__main__":
    main()