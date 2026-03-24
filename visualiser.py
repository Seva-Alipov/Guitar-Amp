#AI Generated for debugging purposes. Not used due to USART overhead making it unusable.

import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ── Config ────────────────────────────────────────────
PORT        = '/dev/ttyACM0'
BAUD        = 115200
SAMPLE_RATE = 3000        # ~7200 samples/sec in binary mode
WINDOW      = 4096        # samples to display and FFT
VREF        = 3.3
ADC_MAX     = 4095

# Note names for frequency → note conversion
NOTE_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note(freq):
    if freq < 20:
        return "---"
    semitones = 12 * np.log2(freq / 440.0)
    midi      = round(semitones) + 69
    octave    = (midi // 12) - 1
    name      = NOTE_NAMES[midi % 12]
    cents     = round((semitones - round(semitones)) * 100)
    return f"{name}{octave} ({freq:.1f} Hz, {cents:+d}c)"

def detect_frequency(samples, sample_rate):
    # Remove DC bias
    samples = samples - np.mean(samples)
    if np.max(np.abs(samples)) < 10:   # silence threshold
        return 0.0
    # FFT
    fft      = np.abs(np.fft.rfft(samples))
    freqs    = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    # Only look in guitar range 70Hz - 1400Hz
    mask     = (freqs >= 70) & (freqs <= 1400)
    fft_masked = fft.copy()
    fft_masked[~mask] = 0
    peak_idx = np.argmax(fft_masked)
    return freqs[peak_idx]

# ── Setup ─────────────────────────────────────────────
ser     = serial.Serial(PORT, BAUD, timeout=1)
buf     = deque([2048] * WINDOW, maxlen=WINDOW)  # pre-fill with mid-scale

fig, (ax_wave, ax_fft) = plt.subplots(2, 1, figsize=(12, 6))
fig.patch.set_facecolor('#1e1e1e')
for ax in (ax_wave, ax_fft):
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')

# Waveform plot
x_wave      = np.linspace(0, WINDOW / SAMPLE_RATE * 1000, WINDOW)
line_wave,  = ax_wave.plot(x_wave, list(buf), color='#00ff88', lw=0.8)
ax_wave.set_ylim(0, 4095)
ax_wave.set_xlim(0, x_wave[-1])
ax_wave.set_ylabel('ADC value', color='white')
ax_wave.set_xlabel('Time (ms)', color='white')
ax_wave.axhline(2048, color='#444', lw=0.5, linestyle='--')  # bias line

# FFT plot
x_fft       = np.fft.rfftfreq(WINDOW, 1.0 / SAMPLE_RATE)
line_fft,   = ax_fft.plot(x_fft, np.zeros(len(x_fft)), color='#ff6b6b', lw=0.8)
ax_fft.set_xlim(70, 1400)
ax_fft.set_ylabel('Magnitude', color='white')
ax_fft.set_xlabel('Frequency (Hz)', color='white')

note_text   = ax_fft.text(0.02, 0.85, '---', transform=ax_fft.transAxes,
                           color='#ffff00', fontsize=14, fontweight='bold')
title_text  = fig.suptitle('STM32 ADC Debug Visualizer', color='white', fontsize=13)

plt.tight_layout()

# ── Animation ─────────────────────────────────────────
def read_serial():
    """Read all available bytes and decode 2-byte big-endian samples."""
    raw = ser.read(ser.in_waiting or 2)
    samples = []
    for i in range(0, len(raw) - 1, 2):
        sample = (raw[i] << 8) | raw[i + 1]
        if sample <= ADC_MAX:
            samples.append(sample)
    return samples

def update(_frame):
    new_samples = read_serial()
    buf.extend(new_samples)

    data     = np.array(buf, dtype=np.float32)
    line_wave.set_ydata(data)

    # FFT
    fft_mag  = np.abs(np.fft.rfft(data - np.mean(data)))
    line_fft.set_ydata(fft_mag)
    ax_fft.set_ylim(0, np.max(fft_mag) * 1.1 + 1)

    # Note detection
    freq     = detect_frequency(data, SAMPLE_RATE)
    note_text.set_text(freq_to_note(freq))

    return line_wave, line_fft, note_text

ani = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)

try:
    plt.show()
finally:
    ser.close()