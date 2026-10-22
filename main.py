
import subprocess
import sys
import time
from pathlib import Path

file_path = Path(
r"file_path.wav"
)

# Prepare waveform process first
start_time = time.time() + 1.5

visual = subprocess.Popen([
    sys.executable,
    "display_waveform.py",
    str(file_path),
    str(start_time)
])

# Start playback close to visualization start
delay = start_time - time.time()
if delay > 0:
    time.sleep(delay)

audio = subprocess.Popen([
    sys.executable,
    "play_audio.py",
    str(file_path)
])

audio.wait()
visual.wait()



