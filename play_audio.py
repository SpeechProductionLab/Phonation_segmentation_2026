import soundfile as sf
import sounddevice as sd
import time
import sys

file_path = sys.argv[1]

audio, sr = sf.read(
    file_path,
    dtype="float32",
    always_2d=True
)

# time.sleep(0.4)

sd.play(audio, sr)
sd.wait()


