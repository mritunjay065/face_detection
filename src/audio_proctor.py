"""
Audio Proctoring Engine using pure Python sounddevice & NumPy.
Features:
- Energy & Spectral Voice Activity Detection (VAD)
- Detects continuous background talking, whispering, or coaching
- Zero-latency non-blocking background audio ingestion
- Real-time decibel (dB) and speech probability metrics
"""
import threading
import time
from typing import Dict, Optional
import numpy as np


class AudioProctoringEngine:
    def __init__(self, sample_rate: int = 16000, block_size: int = 1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.running = True
        self.stream = None

        # Energy & Speech Detection thresholds
        # Standard quiet room: -50dB to -40dB. Whispering / Talking: > -30dB
        self.speech_db_threshold = -32.0
        self.voice_activity_count = 0
        self.continuous_speech_seconds = 0.0

        self.lock = threading.Lock()
        self.cached_results = {
            "is_talking": False,
            "current_db": -60.0,
            "speech_alert": False,
            "continuous_speech_sec": 0.0,
            "audio_violation": False,
        }

        self._start_audio_stream()

    def _audio_callback(self, indata, frames, time_info, status):
        """Non-blocking low-latency callback called by sounddevice driver."""
        if not self.running:
            return

        # indata is numpy array of shape (block_size, channels)
        audio_data = indata[:, 0]
        # Calculate RMS energy in decibels
        rms = np.sqrt(np.mean(audio_data ** 2) + 1e-12)
        db = float(20.0 * np.log10(max(1e-5, rms)))

        # Frequency domain check for human voice fundamental range (85 Hz to 255 Hz)
        fft_vals = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(len(audio_data), 1.0 / self.sample_rate)

        voice_band = (freqs >= 85) & (freqs <= 350)
        voice_energy = float(np.sum(fft_vals[voice_band]))
        total_energy = float(np.sum(fft_vals) + 1e-8)
        voice_ratio = voice_energy / total_energy

        # Human speech signature: energy above quiet floor and significant voice band ratio
        is_speech = (db > self.speech_db_threshold) and (voice_ratio > 0.15)

        with self.lock:
            if is_speech:
                self.continuous_speech_seconds += (self.block_size / self.sample_rate)
            else:
                self.continuous_speech_seconds = max(0.0, self.continuous_speech_seconds - 0.15)

            # Flag violation if sustained talking or whispering lasts > 1.2 seconds
            speech_violation = self.continuous_speech_seconds > 1.2

            self.cached_results = {
                "is_talking": is_speech,
                "current_db": round(db, 1),
                "speech_alert": is_speech,
                "continuous_speech_sec": round(self.continuous_speech_seconds, 1),
                "audio_violation": speech_violation,
            }

    def _start_audio_stream(self):
        try:
            import sounddevice as sd
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback
            )
            self.stream.start()
            print("[Info] Audio Proctoring Engine started (Listening on default microphone).")
        except Exception as e:
            print(f"[Warning] Audio Proctoring stream could not start: {e}")

    def get_audio_state(self) -> Dict:
        """Instant non-blocking retrieval of audio security telemetry."""
        with self.lock:
            return dict(self.cached_results)

    def close(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
