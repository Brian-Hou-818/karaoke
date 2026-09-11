import sys
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt_zi, sosfilt

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox, QComboBox, QPushButton, QGroupBox
)

# ==========================================
# AUDIO CONFIGURATION & DSP WORKER
# ==========================================
SAMPLE_RATE = 44100
BLOCK_SIZE = 512


class AudioEngineThread(QThread):
    level_signal = pyqtSignal(float)  # For live peak meter update

    def __init__(self, input_device=None, output_device=None):
        super().__init__()
        self.input_device = input_device
        self.output_device = output_device
        self.running = False

        # Adjustable DSP Parameters
        self.gain = 3.5
        self.hp_cutoff = 100.0
        self.hp_enabled = True
        self.reverb_enabled = False
        self.reverb_decay = 0.4
        self.reverb_delay_ms = 120

        # High-Pass Filter Setup
        self._update_filter()

        # Simple Reverb Delay Line Buffer
        self.delay_samples = int(SAMPLE_RATE * (self.reverb_delay_ms / 1000.0))
        self.delay_buffer = np.zeros((self.delay_samples, 2), dtype=np.float32)
        self.delay_ptr = 0

    def _update_filter(self):
        """Re-calculates high-pass filter coefficients."""
        self.sos = butter(2, self.hp_cutoff, 'hp', fs=SAMPLE_RATE, output='sos')
        self.zi = sosfilt_zi(self.sos)
        self.filter_state = None

    def set_hp_cutoff(self, cutoff):
        self.hp_cutoff = cutoff
        self._update_filter()

    def set_gain(self, gain_val):
        self.gain = gain_val

    def set_hp_enabled(self, enabled):
        self.hp_enabled = enabled

    def set_reverb(self, enabled, decay=0.4, delay_ms=120):
        self.reverb_enabled = enabled
        self.reverb_decay = decay
        self.reverb_delay_ms = delay_ms
        self.delay_samples = int(SAMPLE_RATE * (self.reverb_delay_ms / 1000.0))
        self.delay_buffer = np.zeros((self.delay_samples, 2), dtype=np.float32)
        self.delay_ptr = 0

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio Warning: {status}", file=sys.stderr)

        channels = indata.shape[1]

        # 1. State initialization for filter continuity
        if self.filter_state is None or self.filter_state.shape[1] != channels:
            self.filter_state = np.repeat(self.zi[:, np.newaxis], channels, axis=1)

        # 2. High-Pass Filter (Strip static-causing sub-100Hz rumble)
        if self.hp_enabled:
            processed, self.filter_state = sosfilt(self.sos, indata, axis=0, zi=self.filter_state)
        else:
            processed = indata.copy()

        # 3. Apply Digital Gain Boost
        processed = processed * self.gain

        # 4. Optional Vocal Reverb/Echo Processing
        if self.reverb_enabled:
            for i in range(frames):
                delay_idx = (self.delay_ptr + i) % self.delay_samples
                echo = self.delay_buffer[delay_idx, :channels]

                # Combine input audio with decaying feedback delay
                current_sample = processed[i, :] + (echo * self.reverb_decay)
                self.delay_buffer[delay_idx, :channels] = current_sample
                processed[i, :] = current_sample

            self.delay_ptr = (self.delay_ptr + frames) % self.delay_samples

        # 5. Peak Meter Signal Emission
        peak = np.max(np.abs(processed))
        self.level_signal.emit(float(peak))

        # 6. Output Clamping [-1.0 to 1.0] to prevent digital distortion
        outdata[:] = np.clip(processed, -1.0, 1.0)

    def run(self):
        self.running = True
        try:
            in_info = sd.query_devices(self.input_device, 'input')
            out_info = sd.query_devices(self.output_device, 'output')
            channels = min(in_info['max_input_channels'], out_info['max_output_channels'], 2)

            stream_kwargs = {
                'channels': channels,
                'samplerate': SAMPLE_RATE,
                'blocksize': BLOCK_SIZE,
                'dtype': 'float32',
                'callback': self.audio_callback
            }

            if self.input_device == self.output_device:
                stream_ctx = sd.Stream(device=self.input_device, **stream_kwargs)
            else:
                stream_ctx = sd.Stream(device=(self.input_device, self.output_device), **stream_kwargs)

            with stream_ctx:
                while self.running:
                    self.msleep(100)

        except Exception as e:
            print(f"Audio Engine Error: {e}", file=sys.stderr)

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# PYQT6 CONTROL INTERFACE
# ==========================================
class KTVVocalWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KTV Vocal Engine & Audio Passthrough")
        self.setGeometry(200, 200, 500, 450)

        self.audio_thread = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Device Selection Group
        device_group = QGroupBox("Hardware Audio Routing")
        device_layout = QVBoxLayout()

        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.populate_devices()

        device_layout.addWidget(QLabel("Microphone Input:"))
        device_layout.addWidget(self.input_combo)
        device_layout.addWidget(QLabel("Audio Output:"))
        device_layout.addWidget(self.output_combo)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Processing Group
        dsp_group = QGroupBox("Microphone DSP Settings")
        dsp_layout = QVBoxLayout()

        # Gain Control Slider
        self.gain_label = QLabel("Digital Gain: 3.5x")
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(10, 80)  # 1.0x to 8.0x
        self.gain_slider.setValue(35)
        self.gain_slider.valueChanged.connect(self.update_dsp)

        # High Pass Filter Toggle & Cutoff
        self.hp_check = QCheckBox("Enable High-Pass Filter (100 Hz Cutoff)")
        self.hp_check.setChecked(True)
        self.hp_check.toggled.connect(self.update_dsp)

        # Reverb Toggle
        self.reverb_check = QCheckBox("Enable Vocal Reverb/Echo")
        self.reverb_check.setChecked(False)
        self.reverb_check.toggled.connect(self.update_dsp)

        dsp_layout.addWidget(self.gain_label)
        dsp_layout.addWidget(self.gain_slider)
        dsp_layout.addWidget(self.hp_check)
        dsp_layout.addWidget(self.reverb_check)
        dsp_group.setLayout(dsp_layout)
        layout.addWidget(dsp_group)

        # Peak Level Meter Display
        meter_group = QGroupBox("Live Output Level")
        meter_layout = QVBoxLayout()
        self.meter_label = QLabel("Level: [          ] 0.00")
        meter_layout.addWidget(self.meter_label)
        meter_group.setLayout(meter_layout)
        layout.addWidget(meter_group)

        # Engine Toggle Button
        self.start_btn = QPushButton("Start Vocal Passthrough")
        self.start_btn.setFixedHeight(40)
        self.start_btn.clicked.connect(self.toggle_engine)
        layout.addWidget(self.start_btn)

    def populate_devices(self):
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.input_combo.addItem(f"{i}: {dev['name']}", i)
            if dev['max_output_channels'] > 0:
                self.output_combo.addItem(f"{i}: {dev['name']}", i)

    def update_dsp(self):
        gain_val = self.gain_slider.value() / 10.0
        self.gain_label.setText(f"Digital Gain: {gain_val:.1f}x")

        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.set_gain(gain_val)
            self.audio_thread.set_hp_enabled(self.hp_check.isChecked())
            self.audio_thread.set_reverb(self.reverb_check.isChecked())

    def update_meter(self, peak):
        bars = int(peak * 20)
        bars = min(bars, 20)
        meter_str = "|" * bars + " " * (20 - bars)
        self.meter_label.setText(f"Level: [{meter_str}] {peak:.2f}")

    def toggle_engine(self):
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread = None
            self.start_btn.setText("Start Vocal Passthrough")
            self.start_btn.setStyleSheet("")
            self.input_combo.setEnabled(True)
            self.output_combo.setEnabled(True)
        else:
            in_dev = self.input_combo.currentData()
            out_dev = self.output_combo.currentData()

            self.audio_thread = AudioEngineThread(input_device=in_dev, output_device=out_dev)
            self.audio_thread.level_signal.connect(self.update_meter)
            self.update_dsp()
            self.audio_thread.start()

            self.start_btn.setText("Stop Vocal Passthrough")
            self.start_btn.setStyleSheet("background-color: #d9534f; color: white;")
            self.input_combo.setEnabled(False)
            self.output_combo.setEnabled(False)

    def closeEvent(self, event):
        if self.audio_thread:
            self.audio_thread.stop()
        event.accept()


# ==========================================
# MAIN APPLICATION ENTRY
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KTVVocalWindow()
    window.show()
    sys.exit(app.exec())