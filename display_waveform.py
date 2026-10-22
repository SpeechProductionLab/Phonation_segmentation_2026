
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys
import time

file_path = sys.argv[1]
start_time = float(sys.argv[2])

audio, sr = sf.read(file_path, dtype="float32", always_2d=True)

waveform = audio.mean(axis=1)
duration = len(audio) / sr
times = np.arange(len(audio)) / sr

fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(times, waveform, color="steelblue", linewidth=0.5)
cursor = ax.axvline(0, color="red", linewidth=2)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")
ax.set_title("Speech Waveform")
ax.set_xlim(0, duration)
ax.grid(True)

def update():
    elapsed = min(max(time.time() - start_time - 0.7, 0), duration)
    cursor.set_xdata([elapsed, elapsed])
    fig.canvas.draw_idle()
    if elapsed >= duration:
        timer.stop()

timer = fig.canvas.new_timer(interval=100)
timer.add_callback(update)
timer.start()

plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# import soundfile as sf
# import sys
# import time
# from scipy.signal import spectrogram

# file_path = sys.argv[1]
# start_time = float(sys.argv[2])

# SYNC_OFFSET = 0.4

# audio, sr = sf.read(file_path, dtype="float32", always_2d=True)
# waveform = audio.mean(axis=1)
# duration = len(waveform) / sr
# times = np.arange(len(waveform)) / sr

# # Calculate spectrogram
# nperseg = min(1024, len(waveform))
# frequencies, spec_times, power = spectrogram(
#     waveform,
#     fs=sr,
#     nperseg=nperseg,
#     noverlap=nperseg * 3 // 4
# )
# power_db = 10 * np.log10(power + 1e-12)

# # Two panels
# fig, (ax1, ax2) = plt.subplots(
#     2, 1, figsize=(16, 8),
#     sharex=True,
#     gridspec_kw={"height_ratios": [2, 1]},
#     constrained_layout=True
# )

# # Detailed waveform
# ax1.plot(times, waveform, color="steelblue", linewidth=0.6)
# ax1.set_ylabel("Amplitude")
# ax1.set_title("Speech Waveform")
# ax1.grid(True, alpha=0.3)

# # Spectrogram
# ax2.pcolormesh(
#     spec_times,
#     frequencies,
#     power_db,
#     shading="auto",
#     cmap="magma"
# )
# ax2.set_ylim(0, min(5000, sr / 2))
# ax2.set_ylabel("Frequency (Hz)")
# ax2.set_xlabel("Time (seconds)")

# # Synchronized playback cursors
# cursor1 = ax1.axvline(0, color="red", linewidth=1.5)
# cursor2 = ax2.axvline(0, color="cyan", linewidth=1.5)

# # Start by displaying the full recording
# ax1.set_xlim(0, duration)

# def update():
#     elapsed = min(
#         max(time.time() - start_time - SYNC_OFFSET, 0),
#         duration
#     )
#     cursor1.set_xdata([elapsed, elapsed])
#     cursor2.set_xdata([elapsed, elapsed])
#     fig.canvas.draw_idle()

#     if elapsed >= duration:
#         timer.stop()

# # Zoom with mouse scroll
# def on_scroll(event):
#     if event.inaxes not in (ax1, ax2):
#         return

#     xmin, xmax = ax1.get_xlim()
#     center = event.xdata
#     if center is None:
#         return

#     scale = 0.7 if event.button == "up" else 1.3
#     new_width = min((xmax - xmin) * scale, duration)
#     new_width = max(new_width, min(0.01, duration))

#     left = center - (center - xmin) * new_width / (xmax - xmin)
#     left = max(0, min(left, duration - new_width))

#     ax1.set_xlim(left, left + new_width)
#     fig.canvas.draw_idle()

# fig.canvas.mpl_connect("scroll_event", on_scroll)

# timer = fig.canvas.new_timer(interval=100)
# timer.add_callback(update)
# timer.start()

# plt.show()



