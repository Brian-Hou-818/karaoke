import ctypes
import os
import random
import sys
import threading
from pathlib import Path
from queue import Queue

# Ensure Qt uses X11 (xcb) on Linux to support VLC window embedding
if not sys.platform.startswith("win"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from flask import Flask, jsonify, render_template_string, request
import numpy as np

# SciPy DSP for Parametric EQ calculation
from scipy.signal import butter, iirnotch, sosfilt
import sounddevice as sd

# ==============================================================================
# 1. WINDOWS / LINUX DLL PATCH & SETUP FOR PYTHON-VLC
# ==============================================================================
VLC_PATH = r"C:\Program Files\VideoLAN\VLC"

if os.path.exists(VLC_PATH):
    os.add_dll_directory(VLC_PATH)
    os.environ["PATH"] = VLC_PATH + os.pathsep + os.environ["PATH"]
    os.environ["PYTHON_VLC_MODULE_PATH"] = VLC_PATH

_orig_cdll_init = ctypes.CDLL.__init__


def _patched_cdll_init(self, name, *args, **kwargs):
    if isinstance(name, str) and ("libvlc" in name or name.startswith(".\\")):
        dll_name = os.path.basename(name)
        abs_vlc_path = os.path.join(VLC_PATH, dll_name)
        if os.path.exists(abs_vlc_path):
            name = abs_vlc_path
        kwargs["winmode"] = 0
    _orig_cdll_init(self, name, *args, **kwargs)


ctypes.CDLL.__init__ = _patched_cdll_init

import vlc

ctypes.CDLL.__init__ = _orig_cdll_init

# ==============================================================================
# 2. APPLICATION IMPORTS
# ==============================================================================
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def get_truncated_title(title, max_length=10):
    if len(title) > max_length:
        return title[: max_length - 3] + "..."
    return title


# ==============================================================================
# 3. DSP PARAMETRIC EQUALIZER FUNCTION
# ==============================================================================
def apply_parametric_vocal_eq(audio_data, sample_rate=44100):
    if audio_data.size == 0:
        return audio_data

    processed = audio_data.copy()

    # 1. High-Pass Filter (< 100 Hz cut)
    hp_sos = butter(2, 100, btype="highpass", fs=sample_rate, output="sos")
    processed = sosfilt(hp_sos, processed, axis=0)

    # 2. Low-Pass Filter (> 8,000 Hz cut)
    lp_sos = butter(2, 8000, btype="lowpass", fs=sample_rate, output="sos")
    processed = sosfilt(lp_sos, processed, axis=0)

    # 3. Parametric Vocal Notch Band Cuts (80 Hz - 4,000 Hz core vocal range)
    vocal_center_frequencies = [250, 800, 1500, 3000]
    wide_q_factor = 0.85

    for freq in vocal_center_frequencies:
        b, a = iirnotch(freq, wide_q_factor, fs=sample_rate)
        if processed.ndim > 1:
            for ch in range(processed.shape[1]):
                processed[:, ch] = np.convolve(
                    processed[:, ch], b, mode="same"
                )
        else:
            processed = np.convolve(processed, b, mode="same")

    return np.clip(processed * 0.3, -1.0, 1.0)


# ==============================================================================
# 4. LOW-LATENCY SOFTWARE AUDIO STREAM WITH ECHO / REVERB DSP
# ==============================================================================
class AudioPassthroughStream:
    def __init__(self, input_device_id, output_device_id=None, sample_rate=44100):
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.sample_rate = sample_rate
        self.volume = 1.0

        self.echo_delay_ms = 180
        self.echo_feedback = 0.4

        self.buffer_size = sample_rate * 2
        self.delay_buffer = np.zeros((self.buffer_size, 2), dtype=np.float32)
        self.write_pos = 0

        self.is_running = False
        self.stream = None

    def _audio_callback(self, indata, outdata, frames, time, status):
        out_channels = outdata.shape[1]

        if indata.shape[1] == 1 and out_channels == 2:
            in_samples = np.column_stack((indata[:, 0], indata[:, 0]))
        else:
            in_samples = indata[:, :out_channels]

        processed = in_samples * self.volume

        delay_samples = int((self.echo_delay_ms / 1000.0) * self.sample_rate)
        read_indices = (
            np.arange(self.write_pos, self.write_pos + frames) - delay_samples
        ) % self.buffer_size
        write_indices = (
            np.arange(self.write_pos, self.write_pos + frames)
        ) % self.buffer_size

        delayed_samples = self.delay_buffer[read_indices, :out_channels]
        mixed_samples = processed + (delayed_samples * self.echo_feedback)

        self.delay_buffer[write_indices, :out_channels] = mixed_samples
        self.write_pos = (self.write_pos + frames) % self.buffer_size

        outdata[:] = np.clip(mixed_samples, -1.0, 1.0)

    def start(self):
        if self.is_running:
            return
        try:
            dev_info = sd.query_devices(self.input_device_id)
            channels = min(dev_info["max_input_channels"], 2)

            device_tuple = (self.input_device_id, self.output_device_id)

            self.stream = sd.Stream(
                device=device_tuple,
                samplerate=self.sample_rate,
                blocksize=64,
                latency="low",
                channels=channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.is_running = True
        except Exception:
            try:
                device_tuple = (self.input_device_id, self.output_device_id)
                self.stream = sd.Stream(
                    device=device_tuple,
                    samplerate=self.sample_rate,
                    blocksize=128,
                    latency="low",
                    channels=channels,
                    dtype="float32",
                    callback=self._audio_callback,
                )
                self.stream.start()
                self.is_running = True
            except Exception as inner_e:
                print(f"Failed to start low-latency mic stream: {inner_e}")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False

    def set_volume(self, level_0_to_1):
        self.volume = level_0_to_1

    def set_echo_params(self, delay_ms, feedback):
        self.echo_delay_ms = delay_ms
        self.echo_feedback = feedback


# ==============================================================================
# 5. DRAG AND DROP QUEUE WIDGET
# ==============================================================================
class DraggableQueueList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.reorder_callback = None

    def dropEvent(self, event):
        super().dropEvent(event)
        if self.reorder_callback:
            self.reorder_callback()


# ==============================================================================
# 6. DEDICATED SECONDARY VIDEO DISPLAY WINDOW
# ==============================================================================
class VideoDisplayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KTV Video Display")
        self.setGeometry(900, 80, 1024, 576)
        self.setStyleSheet("background-color: #000000;")
        self.skip_callback = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_frame = QFrame(self)
        self.video_frame.setStyleSheet("background-color: #000000;")
        layout.addWidget(self.video_frame)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() in (
            Qt.Key.Key_N,
            Qt.Key.Key_Right,
            Qt.Key.Key_MediaNext,
        ):
            if self.skip_callback:
                self.skip_callback()
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()


# ==============================================================================
# 7. MAIN KTV CONTROL WINDOW
# ==============================================================================
class KTVControlWindow(QMainWindow):
    # Qt Signal for thread-safe UI updates from Flask remote
    web_song_added = pyqtSignal(dict)

    def __init__(self, display_window):
        super().__init__()
        self.display_win = display_window
        self.display_win.skip_callback = self.play_next

        self.setWindowTitle("KTV Control Panel")
        self.setGeometry(80, 80, 950, 950)

        self.current_folder = str(Path.home() / "Downloads")

        self.song_library = []
        self.selected_queue = []
        self.current_song = None
        self.vocal_eq_active = False

        self.mic1_stream = None
        self.mic2_stream = None

        self.vlc_instance = vlc.Instance(
            "--aout=directsound" if sys.platform.startswith("win") else ""
        )
        self.media_player = self.vlc_instance.media_player_new()

        # Connect VLC video output to VideoDisplayWindow frame with int handle conversion
        window_handle = int(self.display_win.video_frame.winId())
        if sys.platform.startswith("win"):
            self.media_player.set_hwnd(window_handle)
        else:
            self.media_player.set_xwindow(window_handle)

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(500)
        self.poll_timer.timeout.connect(self.check_media_status)
        self.poll_timer.start()

        self.web_song_added.connect(self.add_song_to_queue)

        self.init_ui()
        self.setup_shortcuts()
        self.populate_audio_devices()

        self.scan_folder_for_songs(self.current_folder)

    def setup_shortcuts(self):
        self.shortcut_n = QShortcut(QKeySequence("N"), self)
        self.shortcut_n.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_n.activated.connect(self.play_next)

        self.shortcut_ctrl_right = QShortcut(
            QKeySequence("Ctrl+Right"), self
        )
        self.shortcut_ctrl_right.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.shortcut_ctrl_right.activated.connect(self.play_next)

        self.shortcut_media_next = QShortcut(
            QKeySequence(Qt.Key.Key_MediaNext), self
        )
        self.shortcut_media_next.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.shortcut_media_next.activated.connect(self.play_next)

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; }
            QLabel { color: #f8fafc; font-size: 13px; }
            QGroupBox {
                color: #38bdf8;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #334155;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 12px;
                background-color: #1e293b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
            }
            QComboBox { 
                background-color: #0f172a; color: white; 
                border: 1px solid #334155; border-radius: 6px; padding: 6px;
                font-size: 12px;
            }
            QLineEdit { 
                background-color: #1e293b; color: white; 
                border: 1px solid #334155; border-radius: 6px; 
                padding: 8px; font-size: 13px;
            }
            QListWidget { 
                background-color: #1e293b; color: #f8fafc; 
                border: 1px solid #334155; border-radius: 8px; padding: 4px;
                font-size: 13px;
            }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #334155; }
            QListWidget::item:hover { background-color: #334155; }
            QListWidget::item:selected { background-color: #2563eb; border-radius: 4px; }
            QPushButton { 
                background-color: #334155; color: white; border: none; 
                padding: 8px 14px; border-radius: 6px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #475569; }
            QPushButton#primaryBtn { background-color: #2563eb; }
            QPushButton#primaryBtn:hover { background-color: #1d4ed8; }
            QPushButton#refreshBtn { background-color: #0284c7; }
            QPushButton#refreshBtn:hover { background-color: #0369a1; }
            QPushButton#playBtn { background-color: #16a34a; min-width: 90px; }
            QPushButton#playBtn:hover { background-color: #15803d; }
            QPushButton#micBtnOff { background-color: #dc2626; min-width: 90px; }
            QPushButton#micBtnOff:hover { background-color: #b91c1c; }
            QPushButton#micBtnOn { background-color: #16a34a; min-width: 90px; }
            QPushButton#micBtnOn:hover { background-color: #15803d; }
            QPushButton#eqFilterOff { background-color: #475569; }
            QPushButton#eqFilterOn { background-color: #059669; }
            QPushButton#eqFilterOn:hover { background-color: #047857; }
            QCheckBox { color: #f8fafc; font-weight: bold; font-size: 12px; }
            QCheckBox::indicator { width: 18px; height: 18px; }

            QSlider::groove:horizontal {
                border: 1px solid #334155; height: 10px; 
                background: #0f172a; border-radius: 5px;
            }
            QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 5px; }
            QSlider::handle:horizontal {
                background: #f8fafc; width: 20px; 
                margin-top: -5px; margin-bottom: -5px; border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #60a5fa;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(16, 16, 16, 16)

        top_bar = QHBoxLayout()
        self.now_playing_label = QLabel("<b>Now Playing:</b> None")
        self.now_playing_label.setStyleSheet(
            "color: #38bdf8; font-size: 15px;"
        )
        top_bar.addWidget(self.now_playing_label, stretch=1)

        self.btn_fullscreen = QPushButton("🖥️ Display Window (F)")
        self.btn_fullscreen.clicked.connect(self.toggle_display_fullscreen)
        top_bar.addWidget(self.btn_fullscreen)

        self.chk_autoplay = QCheckBox("🔄 Autoplay Queue")
        self.chk_autoplay.setChecked(True)
        top_bar.addWidget(self.chk_autoplay)

        self.chk_auto_random = QCheckBox("🎲 Auto-Random Fallback")
        self.chk_auto_random.setChecked(True)
        top_bar.addWidget(self.chk_auto_random)

        root_layout.addLayout(top_bar)

        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(16)

        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        out_box = QGroupBox("🔊 Output Hardware Device")
        out_layout = QVBoxLayout(out_box)
        out_layout.setSpacing(10)

        out_dev_row = QHBoxLayout()
        out_dev_row.addWidget(QLabel("<b>Output:</b>"))
        self.output_combo = QComboBox()
        self.output_combo.currentIndexChanged.connect(
            self.on_output_device_changed
        )
        out_dev_row.addWidget(self.output_combo, stretch=1)
        out_layout.addLayout(out_dev_row)
        left_col.addWidget(out_box)

        music_box = QGroupBox("🎵 Music Playback & Audio Control")
        music_layout = QVBoxLayout(music_box)
        music_layout.setSpacing(10)

        folder_row = QHBoxLayout()
        self.btn_open = QPushButton("📁 Change Folder")
        self.btn_open.setObjectName("primaryBtn")
        self.btn_open.clicked.connect(self.select_folder)
        folder_row.addWidget(self.btn_open)

        self.btn_refresh = QPushButton("🔄 Refresh Library")
        self.btn_refresh.setObjectName("refreshBtn")
        self.btn_refresh.clicked.connect(self.refresh_library)
        folder_row.addWidget(self.btn_refresh)
        music_layout.addLayout(folder_row)

        playback_row = QHBoxLayout()
        self.btn_play_pause = QPushButton("⏸ Pause")
        self.btn_play_pause.setObjectName("playBtn")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        playback_row.addWidget(self.btn_play_pause)

        self.btn_skip = QPushButton("⏭ Skip Song (N)")
        self.btn_skip.clicked.connect(self.play_next)
        playback_row.addWidget(self.btn_skip)
        music_layout.addLayout(playback_row)

        track_row = QHBoxLayout()
        track_row.addWidget(QLabel("<b>Track Mode:</b>"))
        self.btn_stereo = QPushButton("Stereo")
        self.btn_stereo.clicked.connect(
            lambda: self.set_audio_channel("stereo")
        )
        track_row.addWidget(self.btn_stereo)

        self.btn_left = QPushButton("Music (L)")
        self.btn_left.clicked.connect(lambda: self.set_audio_channel("left"))
        track_row.addWidget(self.btn_left)

        self.btn_right = QPushButton("Vocal (R)")
        self.btn_right.clicked.connect(lambda: self.set_audio_channel("right"))
        track_row.addWidget(self.btn_right)
        music_layout.addLayout(track_row)

        eq_row = QHBoxLayout()
        self.btn_vocal_eq = QPushButton(
            "🎙️ Vocal Parametric EQ (80-4kHz Cut): OFF"
        )
        self.btn_vocal_eq.setObjectName("eqFilterOff")
        self.btn_vocal_eq.clicked.connect(self.toggle_vocal_eq)
        eq_row.addWidget(self.btn_vocal_eq)
        music_layout.addLayout(eq_row)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("<b>Music Vol:</b>"))
        self.music_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.music_vol_slider.setRange(0, 100)
        self.music_vol_slider.setValue(80)
        self.music_vol_slider.valueChanged.connect(self.change_music_volume)
        vol_row.addWidget(self.music_vol_slider, stretch=1)
        self.music_vol_label = QLabel("80%")
        self.music_vol_label.setFixedWidth(40)
        vol_row.addWidget(self.music_vol_label)
        music_layout.addLayout(vol_row)

        left_col.addWidget(music_box)

        mic1_box = QGroupBox("🎙️ Microphone 1 Controls")
        mic1_layout = QVBoxLayout(mic1_box)
        mic1_layout.setSpacing(10)

        m1_dev_row = QHBoxLayout()
        self.mic1_combo = QComboBox()
        m1_dev_row.addWidget(self.mic1_combo, stretch=1)
        self.btn_mic1_toggle = QPushButton("Mic 1 OFF")
        self.btn_mic1_toggle.setObjectName("micBtnOff")
        self.btn_mic1_toggle.clicked.connect(self.toggle_mic1)
        m1_dev_row.addWidget(self.btn_mic1_toggle)
        mic1_layout.addLayout(m1_dev_row)

        m1_vol_row = QHBoxLayout()
        m1_vol_row.addWidget(QLabel("Gain Vol:"))
        self.mic1_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic1_vol_slider.setRange(0, 200)
        self.mic1_vol_slider.setValue(100)
        self.mic1_vol_slider.valueChanged.connect(self.update_mic_settings)
        m1_vol_row.addWidget(self.mic1_vol_slider, stretch=1)
        self.mic1_vol_label = QLabel("100%")
        self.mic1_vol_label.setFixedWidth(45)
        m1_vol_row.addWidget(self.mic1_vol_label)
        mic1_layout.addLayout(m1_vol_row)

        left_col.addWidget(mic1_box)

        mic2_box = QGroupBox("🎙️ Microphone 2 Controls")
        mic2_layout = QVBoxLayout(mic2_box)
        mic2_layout.setSpacing(10)

        m2_dev_row = QHBoxLayout()
        self.mic2_combo = QComboBox()
        m2_dev_row.addWidget(self.mic2_combo, stretch=1)
        self.btn_mic2_toggle = QPushButton("Mic 2 OFF")
        self.btn_mic2_toggle.setObjectName("micBtnOff")
        self.btn_mic2_toggle.clicked.connect(self.toggle_mic2)
        m2_dev_row.addWidget(self.btn_mic2_toggle)
        mic2_layout.addLayout(m2_dev_row)

        m2_vol_row = QHBoxLayout()
        m2_vol_row.addWidget(QLabel("Gain Vol:"))
        self.mic2_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic2_vol_slider.setRange(0, 200)
        self.mic2_vol_slider.setValue(100)
        self.mic2_vol_slider.valueChanged.connect(self.update_mic_settings)
        m2_vol_row.addWidget(self.mic2_vol_slider, stretch=1)
        self.mic2_vol_label = QLabel("100%")
        self.mic2_vol_label.setFixedWidth(45)
        m2_vol_row.addWidget(self.mic2_vol_label)
        mic2_layout.addLayout(m2_vol_row)

        left_col.addWidget(mic2_box)

        echo_box = QGroupBox("✨ Master Vocal Echo & Reverb")
        echo_layout = QVBoxLayout(echo_box)
        echo_layout.setSpacing(10)

        delay_row = QHBoxLayout()
        delay_row.addWidget(QLabel("Echo Delay:"))
        self.echo_delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.echo_delay_slider.setRange(50, 500)
        self.echo_delay_slider.setValue(180)
        self.echo_delay_slider.valueChanged.connect(self.update_mic_settings)
        delay_row.addWidget(self.echo_delay_slider, stretch=1)
        self.echo_delay_label = QLabel("180ms")
        self.echo_delay_label.setFixedWidth(45)
        delay_row.addWidget(self.echo_delay_label)
        echo_layout.addLayout(delay_row)

        decay_row = QHBoxLayout()
        decay_row.addWidget(QLabel("Echo Decay:"))
        self.echo_decay_slider = QSlider(Qt.Orientation.Horizontal)
        self.echo_decay_slider.setRange(0, 85)
        self.echo_decay_slider.setValue(40)
        self.echo_decay_slider.valueChanged.connect(self.update_mic_settings)
        decay_row.addWidget(self.echo_decay_slider, stretch=1)
        self.echo_decay_label = QLabel("40%")
        self.echo_decay_label.setFixedWidth(45)
        decay_row.addWidget(self.echo_decay_label)
        echo_layout.addLayout(decay_row)

        left_col.addWidget(echo_box)

        grid_layout.addLayout(left_col, stretch=1)

        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        right_col.addWidget(QLabel("<b>🔍 Browse Song Library</b>"))
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search title or artist...")
        self.search_bar.textChanged.connect(self.filter_songs)
        right_col.addWidget(self.search_bar)

        self.library_list_widget = QListWidget()
        self.library_list_widget.itemDoubleClicked.connect(
            self.add_selected_song_from_browser
        )
        right_col.addWidget(self.library_list_widget, stretch=1)

        self.btn_add_browser = QPushButton("➕ Add Selected to Queue")
        self.btn_add_browser.setObjectName("primaryBtn")
        self.btn_add_browser.clicked.connect(
            self.add_selected_song_from_browser
        )
        right_col.addWidget(self.btn_add_browser)

        right_col.addWidget(
            QLabel("<b>📋 Selected Songs Queue</b> <i>(Drag to reorder)</i>")
        )

        self.queue_widget = DraggableQueueList()
        self.queue_widget.reorder_callback = self.on_queue_reordered
        right_col.addWidget(self.queue_widget, stretch=1)

        btn_remove = QPushButton("❌ Remove Selected")
        btn_remove.clicked.connect(self.remove_from_queue)
        right_col.addWidget(btn_remove)

        grid_layout.addLayout(right_col, stretch=1)

        root_layout.addLayout(grid_layout)

    def toggle_display_fullscreen(self):
        if self.display_win.isFullScreen():
            self.display_win.showNormal()
        else:
            self.display_win.showFullScreen()

    def populate_audio_devices(self):
        self.mic1_combo.clear()
        self.mic2_combo.clear()
        self.output_combo.clear()

        devices = sd.query_devices()
        default_out_idx = sd.default.device[1]

        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                name = f"[{idx}] {dev['name']}"
                self.mic1_combo.addItem(name, userData=idx)
                self.mic2_combo.addItem(name, userData=idx)

            if dev["max_output_channels"] > 0:
                name = f"[{idx}] {dev['name']}"
                self.output_combo.addItem(name, userData=idx)

        if self.mic2_combo.count() > 1:
            self.mic2_combo.setCurrentIndex(1)

        for i in range(self.output_combo.count()):
            if self.output_combo.itemData(i) == default_out_idx:
                self.output_combo.setCurrentIndex(i)
                break

    def get_selected_output_device_id(self):
        return self.output_combo.currentData()

    def on_output_device_changed(self):
        out_id = self.get_selected_output_device_id()

        if self.mic1_stream and self.mic1_stream.is_running:
            self.mic1_stream.stop()
            self.mic1_stream = AudioPassthroughStream(
                self.mic1_stream.input_device_id, out_id
            )
            self.mic1_stream.start()

        if self.mic2_stream and self.mic2_stream.is_running:
            self.mic2_stream.stop()
            self.mic2_stream = AudioPassthroughStream(
                self.mic2_stream.input_device_id, out_id
            )
            self.mic2_stream.start()

        self.update_mic_settings()
        self.sync_vlc_output_device()

    def sync_vlc_output_device(self):
        out_id = self.get_selected_output_device_id()
        if out_id is None:
            return

        target_name = sd.query_devices(out_id)["name"]

        device_enum = self.media_player.audio_output_device_enum()
        if device_enum:
            curr = device_enum
            while curr:
                dev_id = curr.contents.device
                dev_desc = (
                    curr.contents.description.decode("utf-8", errors="ignore")
                    if curr.contents.description
                    else ""
                )
                if (
                    target_name.lower() in dev_desc.lower()
                    or dev_desc.lower() in target_name.lower()
                ):
                    self.media_player.audio_output_device_set(None, dev_id)
                    break
                curr = curr.contents.next

            vlc.libvlc_audio_output_device_list_release(device_enum)

    def toggle_mic1(self):
        if self.mic1_stream and self.mic1_stream.is_running:
            self.mic1_stream.stop()
            self.mic1_stream = None
            self.btn_mic1_toggle.setText("Mic 1 OFF")
            self.btn_mic1_toggle.setObjectName("micBtnOff")
        else:
            dev_id = self.mic1_combo.currentData()
            out_id = self.get_selected_output_device_id()
            if dev_id is not None:
                self.mic1_stream = AudioPassthroughStream(dev_id, out_id)
                self.mic1_stream.start()
                self.btn_mic1_toggle.setText("Mic 1 ON")
                self.btn_mic1_toggle.setObjectName("micBtnOn")
        self.btn_mic1_toggle.setStyle(self.btn_mic1_toggle.style())
        self.update_mic_settings()

    def toggle_mic2(self):
        if self.mic2_stream and self.mic2_stream.is_running:
            self.mic2_stream.stop()
            self.mic2_stream = None
            self.btn_mic2_toggle.setText("Mic 2 OFF")
            self.btn_mic2_toggle.setObjectName("micBtnOff")
        else:
            dev_id = self.mic2_combo.currentData()
            out_id = self.get_selected_output_device_id()
            if dev_id is not None:
                self.mic2_stream = AudioPassthroughStream(dev_id, out_id)
                self.mic2_stream.start()
                self.btn_mic2_toggle.setText("Mic 2 ON")
                self.btn_mic2_toggle.setObjectName("micBtnOn")
        self.btn_mic2_toggle.setStyle(self.btn_mic2_toggle.style())
        self.update_mic_settings()

    def update_mic_settings(self):
        m1_vol = self.mic1_vol_slider.value() / 100.0
        m2_vol = self.mic2_vol_slider.value() / 100.0
        delay_ms = self.echo_delay_slider.value()
        feedback = self.echo_decay_slider.value() / 100.0

        self.mic1_vol_label.setText(f"{self.mic1_vol_slider.value()}%")
        self.mic2_vol_label.setText(f"{self.mic2_vol_slider.value()}%")
        self.echo_delay_label.setText(f"{delay_ms}ms")
        self.echo_decay_label.setText(f"{self.echo_decay_slider.value()}%")

        if self.mic1_stream:
            self.mic1_stream.set_volume(m1_vol)
            self.mic1_stream.set_echo_params(delay_ms, feedback)

        if self.mic2_stream:
            self.mic2_stream.set_volume(m2_vol)
            self.mic2_stream.set_echo_params(delay_ms, feedback)

    def scan_folder_for_songs(self, folder_path):
        if not os.path.exists(folder_path):
            return

        self.current_folder = folder_path
        self.song_library.clear()

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(
                    (".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3")
                ):
                    full_path = os.path.join(root, file)
                    song_name = os.path.splitext(file)[0]
                    self.song_library.append(
                        {"name": song_name, "path": full_path}
                    )

        self.populate_library_list(self.song_library)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select KTV Songs Directory", self.current_folder
        )
        if folder_path:
            self.scan_folder_for_songs(folder_path)

    def refresh_library(self):
        self.scan_folder_for_songs(self.current_folder)

    def populate_library_list(self, song_list):
        self.library_list_widget.clear()
        for song in song_list:
            item = QListWidgetItem(song["name"])
            item.setData(Qt.ItemDataRole.UserRole, song)
            self.library_list_widget.addItem(item)

    def filter_songs(self, text):
        query = text.lower()
        filtered = [
            s for s in self.song_library if query in s["name"].lower()
        ]
        self.populate_library_list(filtered)

    def add_selected_song_from_browser(self):
        selected_items = self.library_list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            song = item.data(Qt.ItemDataRole.UserRole)
            self.add_song_to_queue(song)

    def add_song_to_queue(self, song):
        self.selected_queue.append(song)
        self.refresh_queue_widget()

        if not self.media_player.is_playing() and not self.current_song:
            self.play_next()

    def refresh_queue_widget(self):
        self.queue_widget.clear()
        for song in self.selected_queue:
            item = QListWidgetItem(song["name"])
            item.setData(Qt.ItemDataRole.UserRole, song)
            self.queue_widget.addItem(item)

    def on_queue_reordered(self):
        new_queue = []
        for i in range(self.queue_widget.count()):
            item = self.queue_widget.item(i)
            new_queue.append(item.data(Qt.ItemDataRole.UserRole))
        self.selected_queue = new_queue

    def remove_from_queue(self):
        selected_items = self.queue_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            song = item.data(Qt.ItemDataRole.UserRole)
            if song in self.selected_queue:
                self.selected_queue.remove(song)
        self.refresh_queue_widget()

    def play_next(self):
        if self.selected_queue:
            song = self.selected_queue.pop(0)
            self.refresh_queue_widget()
            self.play_song(song)
        elif self.chk_auto_random.isChecked() and self.song_library:
            random_song = random.choice(self.song_library)
            self.play_song(random_song)
        else:
            self.media_player.stop()
            self.current_song = None
            self.now_playing_label.setText("<b>Now Playing:</b> None")

    def play_song(self, song):
        self.current_song = song
        self.now_playing_label.setText(
            f"<b>Now Playing:</b> {get_truncated_title(song['name'], 40)}"
        )

        media = self.vlc_instance.media_new(song["path"])
        self.media_player.set_media(media)
        self.media_player.play()

        # Re-apply current volume
        self.change_music_volume(self.music_vol_slider.value())

    def toggle_play_pause(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.btn_play_pause.setText("▶ Play")
        else:
            self.media_player.play()
            self.btn_play_pause.setText("⏸ Pause")

    def set_audio_channel(self, mode):
        if mode == "stereo":
            self.media_player.audio_set_channel(vlc.AudioOutputChannel.Stereo)
        elif mode == "left":
            self.media_player.audio_set_channel(vlc.AudioOutputChannel.Left)
        elif mode == "right":
            self.media_player.audio_set_channel(vlc.AudioOutputChannel.Right)

    def toggle_vocal_eq(self):
        self.vocal_eq_active = not self.vocal_eq_active
        if self.vocal_eq_active:
            self.btn_vocal_eq.setText(
                "🎙️ Vocal Parametric EQ (80-4kHz Cut): ON"
            )
            self.btn_vocal_eq.setObjectName("eqFilterOn")
        else:
            self.btn_vocal_eq.setText(
                "🎙️ Vocal Parametric EQ (80-4kHz Cut): OFF"
            )
            self.btn_vocal_eq.setObjectName("eqFilterOff")
        self.btn_vocal_eq.setStyle(self.btn_vocal_eq.style())

    def change_music_volume(self, value):
        self.media_player.audio_set_volume(value)
        self.music_vol_label.setText(f"{value}%")

    def check_media_status(self):
        state = self.media_player.get_state()
        if state == vlc.State.Ended:
            if self.chk_autoplay.isChecked():
                self.play_next()


# ==============================================================================
# 8. FLASK WEB SERVER FOR MOBILE CONTROLLER
# ==============================================================================
app = Flask(__name__)
control_win = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>KTV Mobile Remote</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 15px; background: #0f172a; color: #fff; margin: 0; }
        h2, h3 { text-align: center; margin-top: 10px; color: #38bdf8; }
        input { width: 100%; padding: 12px; box-sizing: border-box; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: white; font-size: 16px; margin-bottom: 15px; }
        ul { list-style: none; padding: 0; margin: 0; }
        li { background: #1e293b; margin-bottom: 8px; padding: 12px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #334155; }
        button { background: #2563eb; color: white; border: none; padding: 8px 14px; border-radius: 6px; font-weight: bold; cursor: pointer; }
        button:active { background: #1d4ed8; }
        .queue-item { background: #0f172a; border-left: 4px solid #38bdf8; }
    </style>
</head>
<body>
    <h2>🎤 KTV Remote</h2>

    <h3>Current Queue</h3>
    <ul id="queueList"></ul>

    <hr style="border-color: #334155; margin: 20px 0;">

    <h3>Song Library</h3>
    <input type="text" id="searchInput" onkeyup="filterSongs()" placeholder="Search library...">
    <ul id="libraryList"></ul>

    <script>
        let fullLibrary = [];

        function fetchQueue() {
            fetch('/api/queue')
                .then(r => r.json())
                .then(data => {
                    const list = document.getElementById('queueList');
                    if (data.queue.length === 0) {
                        list.innerHTML = '<li style="color:#64748b;">Queue is empty</li>';
                    } else {
                        list.innerHTML = data.queue.map((s, i) => `<li class="queue-item"><span>${i+1}. ${s}</span></li>`).join('');
                    }
                });
        }

        function fetchLibrary() {
            fetch('/api/library')
                .then(r => r.json())
                .then(data => {
                    fullLibrary = data.library;
                    renderLibrary(fullLibrary);
                });
        }

        function renderLibrary(songs) {
            const list = document.getElementById('libraryList');
            if (songs.length === 0) {
                list.innerHTML = '<li style="color:#64748b;">No songs found</li>';
                return;
            }
            list.innerHTML = songs.map(song => `
                <li>
                    <span>${song}</span>
                    <button onclick="addSong('${song.replace(/'/g, "\\'")}')">Add</button>
                </li>
            `).join('');
        }

        function filterSongs() {
            const query = document.getElementById('searchInput').value.toLowerCase();
            const filtered = fullLibrary.filter(s => s.toLowerCase().includes(query));
            renderLibrary(filtered);
        }

        function addSong(songTitle) {
            fetch('/api/queue', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ song: songTitle })
            }).then(() => {
                fetchQueue();
            });
        }

        setInterval(fetchQueue, 3000);
        fetchQueue();
        fetchLibrary();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/library", methods=["GET"])
def get_library():
    if control_win:
        titles = [s["name"] for s in control_win.song_library]
        return jsonify({"library": sorted(titles)})
    return jsonify({"library": []})


@app.route("/api/queue", methods=["GET"])
def get_queue():
    if control_win:
        queue_titles = [s["name"] for s in control_win.selected_queue]
        return jsonify({"queue": queue_titles})
    return jsonify({"queue": []})


@app.route("/api/queue", methods=["POST"])
def post_queue():
    data = request.get_json()
    song_name = data.get("song")

    if song_name and control_win:
        matched_song = next(
            (s for s in control_win.song_library if s["name"] == song_name), None
        )
        if matched_song:
            control_win.web_song_added.emit(matched_song)
            return jsonify({"status": "success", "song": song_name}), 200

    return jsonify({"status": "error", "message": "Song not found"}), 400


def start_flask_server():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ==============================================================================
# 9. APPLICATION ENTRY POINT
# ==============================================================================
def main():
    global control_win
    qapp = QApplication(sys.argv)

    display_win = VideoDisplayWindow()
    control_win = KTVControlWindow(display_win)

    server_thread = threading.Thread(target=start_flask_server, daemon=True)
    server_thread.start()

    display_win.show()
    control_win.show()

    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()