import ctypes
import os

os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
os.environ['ALSOFT_DRIVERS'] = 'pulse'

import random
import sys
import threading
from pathlib import Path

# Force PulseAudio/ALSA virtual plugin layers before sounddevice loads
os.environ["PA_ALSA_PLUGHW"] = "1"
os.environ["PORTAUDIO_DISABLE_JACK"] = "1"
# Force Qt to use X11/XCB on Linux for proper VLC video embedding
os.environ["QT_QPA_PLATFORM"] = "xcb"

from flask import Flask, jsonify, render_template_string, request
from flask_socketio import SocketIO, emit
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import sounddevice as sd
import vlc

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def get_truncated_title(title, max_length=40):
    if len(title) > max_length:
        return title[: max_length - 3] + "..."
    return title


# ==============================================================================
# QT SIGNAL EMITTER FOR THREAD-SAFE MIC LEVEL UPDATES
# ==============================================================================
class MicLevelEmitter(QObject):
    level_signal = pyqtSignal(int, bool)


# ==============================================================================
# LOW-LATENCY SOFTWARE AUDIO STREAM WITH SCIPY SOS FILTER & REVERB DSP
# ==============================================================================
class AudioPassthroughStream:

    def __init__(self, input_device_id, output_device_id=None, sample_rate=44100):
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.sample_rate = sample_rate
        self.gain = 2.5
        self.hp_cutoff = 100.0

        self.echo_delay_ms = 180
        self.echo_feedback = 0.20

        self.buffer_size = sample_rate * 2
        self.delay_buffer = np.zeros((self.buffer_size, 2), dtype=np.float32)
        self.write_pos = 0

        self.is_running = False
        self.stream = None
        self._frame_counter = 0

        # High-Pass Filter Setup using Second-Order Sections (SOS)
        self._update_filter()

        self.emitter = MicLevelEmitter()

    def _update_filter(self):
        """Re-calculates high-pass filter coefficients."""
        self.sos = butter(2, self.hp_cutoff, 'hp', fs=self.sample_rate, output='sos')
        self.zi = sosfilt_zi(self.sos)
        self.filter_state = None

    def _audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"[Mic Buffer Warning] {status}", sys.stderr)

        try:
            channels = indata.shape[1]

            # State initialization for filter continuity (shape: n_sections, 2, channels)
            if self.filter_state is None or self.filter_state.shape[2] != channels:
                self.filter_state = np.repeat(self.zi[:, :, np.newaxis], channels, axis=2)

            # High-Pass Filter (Strip static-causing sub-100Hz rumble)
            filtered, self.filter_state = sosfilt(self.sos, indata, axis=0, zi=self.filter_state)

            # Apply Digital Gain Boost (supports up to 10.0x)
            amplified = filtered * self.gain

            # Measure Peak and RMS Levels
            rms = np.sqrt(np.mean(amplified ** 2))
            peak = np.max(np.abs(amplified))

            # Soft Noise Gate Thresholding
            NOISE_GATE_THRESHOLD = 0.004
            if rms < NOISE_GATE_THRESHOLD:
                attenuation = max(0.0, (rms / NOISE_GATE_THRESHOLD) ** 2) if NOISE_GATE_THRESHOLD > 0 else 0
                amplified *= attenuation
                vol_percent = 0
            else:
                vol_percent = int(min(1.0, rms * 4.0) * 100)

            is_clipping = bool(peak > 0.90)

            # Throttle GUI Signal Update
            self._frame_counter += 1
            if self._frame_counter % 10 == 0:
                self.emitter.level_signal.emit(int(vol_percent), is_clipping)
                self._frame_counter = 0

            # Channel Matching
            in_chans = amplified.shape[1]
            out_chans = outdata.shape[1]

            if in_chans == 1 and out_chans >= 2:
                in_samples = np.column_stack((amplified[:, 0], amplified[:, 0]))
            elif in_chans >= 2 and out_chans >= 2:
                in_samples = amplified[:, :2]
            else:
                in_samples = amplified

            processed = np.clip(in_samples * 0.95, -1.0, 1.0)

            # Echo / Delay Loop Buffer
            delay_samples = int((self.echo_delay_ms / 1000.0) * self.sample_rate)
            read_indices = (
                np.arange(self.write_pos, self.write_pos + frames) - delay_samples
            ) % self.buffer_size
            write_indices = (
                np.arange(self.write_pos, self.write_pos + frames)
            ) % self.buffer_size

            delayed_samples = self.delay_buffer[read_indices, : processed.shape[1]]
            mixed_samples = processed + (delayed_samples * self.echo_feedback)

            self.delay_buffer[write_indices, : processed.shape[1]] = mixed_samples * 0.85
            self.write_pos = (self.write_pos + frames) % self.buffer_size

            outdata.fill(0)
            outdata[:, : mixed_samples.shape[1]] = np.clip(mixed_samples, -1.0, 1.0)
        except Exception as e:
            outdata.fill(0)
            print(f"[Callback DSP Error] {e}", sys.stderr)

    def start(self):
        if self.is_running:
            return
        try:
            target_in = (
                self.input_device_id
                if self.input_device_id is not None
                else sd.default.device[0]
            )
            target_out = (
                self.output_device_id
                if self.output_device_id is not None
                else sd.default.device[1]
            )

            in_info = sd.query_devices(target_in, "input")
            out_info = sd.query_devices(target_out, "output")

            in_ch = max(1, int(in_info.get("max_input_channels", 1)))
            out_ch = max(1, min(2, int(out_info.get("max_output_channels", 2))))

            srate = int(in_info.get("default_samplerate", 44100))
            if srate <= 0:
                srate = 44100
            self.sample_rate = srate

            self._update_filter()

            self.buffer_size = self.sample_rate * 2
            self.delay_buffer = np.zeros((self.buffer_size, 2), dtype=np.float32)

            self.stream = sd.Stream(
                device=(target_in, target_out),
                samplerate=self.sample_rate,
                blocksize=256,
                channels=(in_ch, out_ch),
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.is_running = True
            print(f"[Audio Stream] Stream active with High-Pass SOS Filter @ {srate}Hz")
        except Exception as e:
            print(f"[Audio Stream Error] {e}", sys.stderr)
            self.is_running = False

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[Audio Stream Close Error] {e}", sys.stderr)
            self.stream = None
        self.is_running = False

    def set_volume(self, level_0_to_10):
        self.gain = float(level_0_to_10)

    def set_echo_params(self, delay_ms, feedback):
        self.echo_delay_ms = delay_ms
        self.echo_feedback = feedback


# ==============================================================================
# DRAG AND DROP QUEUE WIDGET
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
# SECONDARY VIDEO DISPLAY WINDOW
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
# MAIN KTV CONTROL WINDOW
# ==============================================================================
class KTVControlWindow(QMainWindow):
    web_song_added = pyqtSignal(dict, bool)
    web_action_triggered = pyqtSignal(str)

    def __init__(self, display_window):
        super().__init__()
        self.display_win = display_window
        self.display_win.skip_callback = self.play_next

        self.setWindowTitle("KTV Control Panel (Linux)")
        self.setGeometry(80, 80, 1000, 950)

        self.current_folder = str(Path.home() / "Downloads")

        self.song_library = []
        self.selected_queue = []
        self.current_song = None

        self.playback_rate = 1.0

        self.mic1_stream = None
        self.mic2_stream = None

        vlc_flags = [
            "--no-xlib",
            "--aout=pulse",
            "--role=music",
            "--alsa-audio-device=default",
        ]
        self.vlc_instance = vlc.Instance(" ".join(vlc_flags))
        if self.vlc_instance is None:
            raise RuntimeError("Failed to initialize libVLC instance.")

        self.media_player = self.vlc_instance.media_player_new()

        # Attach handle safely
        window_handle = int(self.display_win.video_frame.winId())
        self.media_player.set_xwindow(window_handle)

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(300)
        self.poll_timer.timeout.connect(self.check_media_status)
        self.poll_timer.start()

        self.web_song_added.connect(self.add_song_to_queue)
        self.web_action_triggered.connect(self.handle_web_action)

        self.init_ui()
        self.setup_shortcuts()
        self.populate_audio_devices()

        self.scan_folder_for_songs(self.current_folder)

    def setup_shortcuts(self):
        self.shortcut_n = QShortcut(QKeySequence("N"), self)
        self.shortcut_n.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_n.activated.connect(self.play_next)

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
            }
            QLineEdit { 
                background-color: #1e293b; color: white; 
                border: 1px solid #334155; border-radius: 6px; padding: 8px;
            }
            QListWidget { 
                background-color: #1e293b; color: #f8fafc; 
                border: 1px solid #334155; border-radius: 8px; padding: 4px;
            }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #334155; }
            QListWidget::item:hover { background-color: #334155; }
            QListWidget::item:selected { background-color: #2563eb; border-radius: 4px; }
            QPushButton { 
                background-color: #334155; color: white; border: none; 
                padding: 8px 14px; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #475569; }
            QPushButton#primaryBtn { background-color: #2563eb; }
            QPushButton#primaryBtn:hover { background-color: #1d4ed8; }
            QPushButton#priorityBtn { background-color: #d97706; }
            QPushButton#priorityBtn:hover { background-color: #b45309; }
            QSlider::groove:horizontal { border: 1px solid #334155; height: 10px; background: #0f172a; border-radius: 5px; }
            QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 5px; }
            QSlider::handle:horizontal { background: #f8fafc; width: 20px; margin-top: -5px; margin-bottom: -5px; border-radius: 10px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)

        top_bar = QHBoxLayout()
        self.now_playing_label = QLabel("<b>Now Playing:</b> None")
        self.now_playing_label.setStyleSheet("color: #38bdf8; font-size: 15px;")
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
        left_col = QVBoxLayout()

        out_box = QGroupBox("🔊 Hardware Audio Output")
        out_layout = QVBoxLayout(out_box)
        self.output_combo = QComboBox()
        self.output_combo.currentIndexChanged.connect(
            self.on_output_device_changed
        )
        out_layout.addWidget(self.output_combo)
        left_col.addWidget(out_box)

        music_box = QGroupBox("🎵 Music & Audio Track Controls")
        music_layout = QVBoxLayout(music_box)

        folder_row = QHBoxLayout()
        self.btn_open = QPushButton("📁 Select Folder")
        self.btn_open.setObjectName("primaryBtn")
        self.btn_open.clicked.connect(self.select_folder)
        folder_row.addWidget(self.btn_open)

        self.btn_refresh = QPushButton("🔄 Refresh Library")
        self.btn_refresh.clicked.connect(self.refresh_library)
        folder_row.addWidget(self.btn_refresh)
        music_layout.addLayout(folder_row)

        playback_row = QHBoxLayout()
        self.btn_play_pause = QPushButton("⏸ Pause")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        playback_row.addWidget(self.btn_play_pause)

        self.btn_skip = QPushButton("⏭ Skip Song (N)")
        self.btn_skip.clicked.connect(self.play_next)
        playback_row.addWidget(self.btn_skip)
        music_layout.addLayout(playback_row)

        tempo_row = QHBoxLayout()
        tempo_row.addWidget(QLabel("<b>Pitch/Speed:</b>"))
        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setRange(80, 120)
        self.tempo_slider.setValue(100)
        self.tempo_slider.valueChanged.connect(self.change_playback_speed)
        tempo_row.addWidget(self.tempo_slider, stretch=1)
        self.tempo_label = QLabel("1.0x")
        tempo_row.addWidget(self.tempo_label)
        music_layout.addLayout(tempo_row)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("<b>Music Vol:</b>"))
        self.music_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.music_vol_slider.setRange(0, 100)
        self.music_vol_slider.setValue(80)
        self.music_vol_slider.valueChanged.connect(self.change_music_volume)
        vol_row.addWidget(self.music_vol_slider, stretch=1)
        self.music_vol_label = QLabel("80%")
        vol_row.addWidget(self.music_vol_label)
        music_layout.addLayout(vol_row)

        left_col.addWidget(music_box)

        mic1_box = QGroupBox("🎙️ Microphone 1 Controls")
        mic1_layout = QVBoxLayout(mic1_box)
        m1_dev_row = QHBoxLayout()
        self.mic1_combo = QComboBox()
        m1_dev_row.addWidget(self.mic1_combo, stretch=1)
        self.btn_mic1_toggle = QPushButton("Mic 1 OFF")
        self.btn_mic1_toggle.clicked.connect(self.toggle_mic1)
        m1_dev_row.addWidget(self.btn_mic1_toggle)
        mic1_layout.addLayout(m1_dev_row)

        m1_vol_row = QHBoxLayout()
        m1_vol_row.addWidget(QLabel("Gain Vol:"))
        self.mic1_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic1_vol_slider.setRange(0, 1000)  # Extended range to 1000% (10x gain)
        self.mic1_vol_slider.setValue(100)
        self.mic1_vol_slider.valueChanged.connect(self.update_mic_settings)
        m1_vol_row.addWidget(self.mic1_vol_slider, stretch=1)
        self.mic1_vol_label = QLabel("100% (1.0x)")
        m1_vol_row.addWidget(self.mic1_vol_label)
        mic1_layout.addLayout(m1_vol_row)

        m1_meter_row = QHBoxLayout()
        m1_meter_row.addWidget(QLabel("Input Level:"))
        self.mic1_bar = QProgressBar()
        self.mic1_bar.setRange(0, 100)
        self.mic1_bar.setValue(0)
        self.mic1_bar.setTextVisible(False)
        self.mic1_bar.setFixedHeight(14)
        self.set_meter_style(self.mic1_bar, False)
        m1_meter_row.addWidget(self.mic1_bar, stretch=1)
        self.mic1_level_label = QLabel("0%")
        m1_meter_row.addWidget(self.mic1_level_label)
        mic1_layout.addLayout(m1_meter_row)

        left_col.addWidget(mic1_box)

        mic2_box = QGroupBox("🎙️ Microphone 2 Controls")
        mic2_layout = QVBoxLayout(mic2_box)
        m2_dev_row = QHBoxLayout()
        self.mic2_combo = QComboBox()
        m2_dev_row.addWidget(self.mic2_combo, stretch=1)
        self.btn_mic2_toggle = QPushButton("Mic 2 OFF")
        self.btn_mic2_toggle.clicked.connect(self.toggle_mic2)
        m2_dev_row.addWidget(self.btn_mic2_toggle)
        mic2_layout.addLayout(m2_dev_row)

        m2_vol_row = QHBoxLayout()
        m2_vol_row.addWidget(QLabel("Gain Vol:"))
        self.mic2_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic2_vol_slider.setRange(0, 1000)  # Extended range to 1000% (10x gain)
        self.mic2_vol_slider.setValue(100)
        self.mic2_vol_slider.valueChanged.connect(self.update_mic_settings)
        m2_vol_row.addWidget(self.mic2_vol_slider, stretch=1)
        self.mic2_vol_label = QLabel("100% (1.0x)")
        m2_vol_row.addWidget(self.mic2_vol_label)
        mic2_layout.addLayout(m2_vol_row)

        m2_meter_row = QHBoxLayout()
        m2_meter_row.addWidget(QLabel("Input Level:"))
        self.mic2_bar = QProgressBar()
        self.mic2_bar.setRange(0, 100)
        self.mic2_bar.setValue(0)
        self.mic2_bar.setTextVisible(False)
        self.mic2_bar.setFixedHeight(14)
        self.set_meter_style(self.mic2_bar, False)
        m2_meter_row.addWidget(self.mic2_bar, stretch=1)
        self.mic2_level_label = QLabel("0%")
        m2_meter_row.addWidget(self.mic2_level_label)
        mic2_layout.addLayout(m2_meter_row)

        left_col.addWidget(mic2_box)

        echo_box = QGroupBox("✨ Master Vocal Echo & Reverb")
        echo_layout = QVBoxLayout(echo_box)
        delay_row = QHBoxLayout()
        delay_row.addWidget(QLabel("Echo Delay:"))
        self.echo_delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.echo_delay_slider.setRange(50, 500)
        self.echo_delay_slider.setValue(180)
        self.echo_delay_slider.valueChanged.connect(self.update_mic_settings)
        delay_row.addWidget(self.echo_delay_slider, stretch=1)
        self.echo_delay_label = QLabel("180ms")
        delay_row.addWidget(self.echo_delay_label)
        echo_layout.addLayout(delay_row)

        decay_row = QHBoxLayout()
        decay_row.addWidget(QLabel("Echo Decay:"))
        self.echo_decay_slider = QSlider(Qt.Orientation.Horizontal)
        self.echo_decay_slider.setRange(0, 85)
        self.echo_decay_slider.setValue(20)
        self.echo_decay_slider.valueChanged.connect(self.update_mic_settings)
        decay_row.addWidget(self.echo_decay_slider, stretch=1)
        self.echo_decay_label = QLabel("20%")
        decay_row.addWidget(self.echo_decay_label)
        echo_layout.addLayout(decay_row)
        left_col.addWidget(echo_box)

        grid_layout.addLayout(left_col, stretch=1)

        right_col = QVBoxLayout()
        self.tab_widget = QTabWidget()

        browse_tab = QWidget()
        browse_layout = QVBoxLayout(browse_tab)
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search title or artist...")
        self.search_bar.textChanged.connect(self.filter_songs)
        browse_layout.addWidget(self.search_bar)

        self.library_list_widget = QListWidget()
        self.library_list_widget.itemDoubleClicked.connect(
            lambda: self.add_selected_song_from_browser(insert_next=False)
        )
        browse_layout.addWidget(self.library_list_widget, stretch=1)

        browse_btn_row = QHBoxLayout()
        self.btn_add_browser = QPushButton("➕ Add to Queue")
        self.btn_add_browser.setObjectName("primaryBtn")
        self.btn_add_browser.clicked.connect(
            lambda: self.add_selected_song_from_browser(insert_next=False)
        )
        browse_btn_row.addWidget(self.btn_add_browser)

        self.btn_insert_browser = QPushButton("⚡ Insert Next (插播)")
        self.btn_insert_browser.setObjectName("priorityBtn")
        self.btn_insert_browser.clicked.connect(
            lambda: self.add_selected_song_from_browser(insert_next=True)
        )
        browse_btn_row.addWidget(self.btn_insert_browser)
        browse_layout.addLayout(browse_btn_row)

        self.tab_widget.addTab(browse_tab, "🔍 Browse Songs")

        suggestions_tab = QWidget()
        suggestions_layout = QVBoxLayout(suggestions_tab)
        self.suggestions_list_widget = QListWidget()
        self.suggestions_list_widget.itemDoubleClicked.connect(
            lambda: self.add_selected_song_from_suggestions(insert_next=False)
        )
        suggestions_layout.addWidget(self.suggestions_list_widget, stretch=1)

        sug_btn_row = QHBoxLayout()
        self.btn_add_suggestion = QPushButton("➕ Add Suggestion")
        self.btn_add_suggestion.setObjectName("primaryBtn")
        self.btn_add_suggestion.clicked.connect(
            lambda: self.add_selected_song_from_suggestions(insert_next=False)
        )
        sug_btn_row.addWidget(self.btn_add_suggestion)

        self.btn_insert_suggestion = QPushButton("⚡ Insert Next (插播)")
        self.btn_insert_suggestion.setObjectName("priorityBtn")
        self.btn_insert_suggestion.clicked.connect(
            lambda: self.add_selected_song_from_suggestions(insert_next=True)
        )
        sug_btn_row.addWidget(self.btn_insert_suggestion)
        suggestions_layout.addLayout(sug_btn_row)

        self.tab_widget.addTab(suggestions_tab, "💡 Song Suggestions")
        right_col.addWidget(self.tab_widget, stretch=1)

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

    def set_meter_style(self, progress_bar, is_clipping=False):
        if is_clipping:
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #444; border-radius: 3px; background-color: #0f172a;
                }
                QProgressBar::chunk { background-color: #ef4444; }
            """)
        else:
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #444; border-radius: 3px; background-color: #0f172a;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #22c55e, stop:0.7 #eab308, stop:1.0 #f97316
                    );
                }
            """)

    def update_mic1_level(self, level, is_clipping):
        self.mic1_bar.setValue(level)
        self.mic1_level_label.setText(f"{level}%")
        self.set_meter_style(self.mic1_bar, is_clipping)

    def update_mic2_level(self, level, is_clipping):
        self.mic2_bar.setValue(level)
        self.mic2_level_label.setText(f"{level}%")
        self.set_meter_style(self.mic2_bar, is_clipping)

    def handle_web_action(self, action):
        if action == "skip":
            self.play_next()
        elif action == "play_pause":
            self.toggle_play_pause()

    def change_playback_speed(self, val):
        self.playback_rate = val / 100.0
        self.tempo_label.setText(f"{self.playback_rate:.1f}x")
        self.media_player.set_rate(self.playback_rate)

    def toggle_display_fullscreen(self):
        if self.display_win.isFullScreen():
            self.display_win.showNormal()
        else:
            self.display_win.showFullScreen()

    def populate_audio_devices(self):
        self.mic1_combo.clear()
        self.mic2_combo.clear()
        self.output_combo.clear()

        try:
            devices = sd.query_devices()
        except Exception as e:
            print(f"[PortAudio Query Warning] {e}", sys.stderr)
            devices = []

        pulse_idx = None
        default_out_idx = None

        try:
            default_out_idx = sd.default.device[1]
        except Exception:
            pass

        for idx, dev in enumerate(devices):
            dev_name = dev["name"].lower()
            if "pulse" in dev_name or "default" in dev_name:
                pulse_idx = idx

            if dev.get("max_input_channels", 0) > 0:
                name = f"[{idx}] {dev['name']}"
                self.mic1_combo.addItem(name, userData=idx)
                self.mic2_combo.addItem(name, userData=idx)

            if dev.get("max_output_channels", 0) > 0:
                name = f"[{idx}] {dev['name']}"
                self.output_combo.addItem(name, userData=idx)

        if pulse_idx is not None:
            for i in range(self.output_combo.count()):
                if self.output_combo.itemData(i) == pulse_idx:
                    self.output_combo.setCurrentIndex(i)
                    break
        elif default_out_idx is not None:
            for i in range(self.output_combo.count()):
                if self.output_combo.itemData(i) == default_out_idx:
                    self.output_combo.setCurrentIndex(i)
                    break

        if self.mic2_combo.count() > 1:
            self.mic2_combo.setCurrentIndex(1)

    def get_selected_output_device_id(self):
        return self.output_combo.currentData()

    def on_output_device_changed(self):
        out_id = self.get_selected_output_device_id()

        if self.mic1_stream and self.mic1_stream.is_running:
            self.mic1_stream.stop()
            dev_id = self.mic1_combo.currentData()
            self.mic1_stream = AudioPassthroughStream(dev_id, out_id)
            self.mic1_stream.emitter.level_signal.connect(self.update_mic1_level)
            self.mic1_stream.start()

        if self.mic2_stream and self.mic2_stream.is_running:
            self.mic2_stream.stop()
            dev_id = self.mic2_combo.currentData()
            self.mic2_stream = AudioPassthroughStream(dev_id, out_id)
            self.mic2_stream.emitter.level_signal.connect(self.update_mic2_level)
            self.mic2_stream.start()

        self.update_mic_settings()

    def toggle_mic1(self):
        if self.mic1_stream and self.mic1_stream.is_running:
            self.mic1_stream.stop()
            self.mic1_stream = None
            self.btn_mic1_toggle.setText("Mic 1 OFF")
            self.update_mic1_level(0, False)
        else:
            dev_id = self.mic1_combo.currentData()
            out_id = self.get_selected_output_device_id()
            self.mic1_stream = AudioPassthroughStream(dev_id, out_id)
            self.mic1_stream.emitter.level_signal.connect(self.update_mic1_level)
            self.mic1_stream.start()
            if self.mic1_stream.is_running:
                self.btn_mic1_toggle.setText("Mic 1 ON")
            else:
                self.btn_mic1_toggle.setText("Mic 1 ERR")
        self.update_mic_settings()

    def toggle_mic2(self):
        if self.mic2_stream and self.mic2_stream.is_running:
            self.mic2_stream.stop()
            self.mic2_stream = None
            self.btn_mic2_toggle.setText("Mic 2 OFF")
            self.update_mic2_level(0, False)
        else:
            dev_id = self.mic2_combo.currentData()
            out_id = self.get_selected_output_device_id()
            self.mic2_stream = AudioPassthroughStream(dev_id, out_id)
            self.mic2_stream.emitter.level_signal.connect(self.update_mic2_level)
            self.mic2_stream.start()
            if self.mic2_stream.is_running:
                self.btn_mic2_toggle.setText("Mic 2 ON")
            else:
                self.btn_mic2_toggle.setText("Mic 2 ERR")
        self.update_mic_settings()

    def update_mic_settings(self):
        m1_vol = self.mic1_vol_slider.value() / 100.0
        m2_vol = self.mic2_vol_slider.value() / 100.0
        delay_ms = self.echo_delay_slider.value()
        feedback = self.echo_decay_slider.value() / 100.0

        self.mic1_vol_label.setText(f"{self.mic1_vol_slider.value()}% ({m1_vol:.1f}x)")
        self.mic2_vol_label.setText(f"{self.mic2_vol_slider.value()}% ({m2_vol:.1f}x)")
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
        self.populate_suggestions_list()

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

    def populate_suggestions_list(self):
        self.suggestions_list_widget.clear()
        suggestions = random.sample(
            self.song_library, min(len(self.song_library), 10)
        )
        for song in suggestions:
            item = QListWidgetItem(f"⭐ {song['name']}")
            item.setData(Qt.ItemDataRole.UserRole, song)
            self.suggestions_list_widget.addItem(item)

    def filter_songs(self, text):
        query = text.lower()
        filtered = [
            s for s in self.song_library if query in s["name"].lower()
        ]
        self.populate_library_list(filtered)

    def add_selected_song_from_browser(self, insert_next=False):
        selected_items = self.library_list_widget.selectedItems()
        for item in selected_items:
            song = item.data(Qt.ItemDataRole.UserRole)
            self.add_song_to_queue(song, insert_next)

    def add_selected_song_from_suggestions(self, insert_next=False):
        selected_items = self.suggestions_list_widget.selectedItems()
        for item in selected_items:
            song = item.data(Qt.ItemDataRole.UserRole)
            self.add_song_to_queue(song, insert_next)

    def add_song_to_queue(self, song, insert_next=False):
        if insert_next:
            self.selected_queue.insert(0, song)
        else:
            self.selected_queue.append(song)

        self.refresh_queue_widget()
        broadcast_system_state()

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
        broadcast_system_state()

    def remove_from_queue(self):
        selected_items = self.queue_widget.selectedItems()
        for item in selected_items:
            song = item.data(Qt.ItemDataRole.UserRole)
            if song in self.selected_queue:
                self.selected_queue.remove(song)
        self.refresh_queue_widget()
        broadcast_system_state()

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

        broadcast_system_state()

    def play_song(self, song):
        self.current_song = song
        self.now_playing_label.setText(
            f"<b>Now Playing:</b> {get_truncated_title(song['name'], 40)}"
        )

        media = self.vlc_instance.media_new(song["path"])
        self.media_player.set_media(media)
        self.media_player.play()

        QTimer.singleShot(200, self._setup_audio_tracks_deferred)

        self.btn_play_pause.setText("⏸ Pause")
        self.change_music_volume(self.music_vol_slider.value())
        self.media_player.set_rate(self.playback_rate)

    def _setup_audio_tracks_deferred(self):
        self.media_player.audio_set_mute(False)
        self.media_player.audio_set_volume(self.music_vol_slider.value())

        track_count = self.media_player.audio_get_track_count()
        if track_count <= 0:
            print("[VLC Warning] No audio tracks parsed or available in file.")
        else:
            self.media_player.audio_set_track(1)

    def toggle_play_pause(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.btn_play_pause.setText("▶ Play")
        else:
            self.media_player.play()
            self.btn_play_pause.setText("⏸ Pause")
        broadcast_system_state()

    def change_music_volume(self, value):
        self.media_player.audio_set_volume(value)
        self.music_vol_label.setText(f"{value}%")

    def check_media_status(self):
        state = self.media_player.get_state()
        if state == vlc.State.Ended:
            if self.chk_autoplay.isChecked():
                self.play_next()

        if self.media_player.is_playing():
            curr_time_ms = self.media_player.get_time()
            total_time_ms = self.media_player.get_length()
            spu_description = self.media_player.video_get_spu_description()

            lyric_line = ""
            if spu_description:
                for spu_id, spu_name in spu_description:
                    if spu_id != -1 and spu_name:
                        lyric_line = spu_name.decode("utf-8", errors="ignore")

            socketio.emit(
                "telemetry",
                {
                    "time_ms": curr_time_ms,
                    "length_ms": total_time_ms,
                    "lyric": lyric_line,
                },
            )

    def closeEvent(self, event):
        if self.mic1_stream:
            self.mic1_stream.stop()
        if self.mic2_stream:
            self.mic2_stream.stop()
        event.accept()


# ==============================================================================
# FLASK WEB SERVER + SOCKET.IO WEBSOCKET MOBILE REMOTE
# ==============================================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = "ktv_secret_key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
control_win = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>KTV Real-Time Mobile Remote</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 15px; background: #0f172a; color: #fff; margin: 0; }
        h2, h3 { text-align: center; margin-top: 10px; color: #38bdf8; }
        .now-playing-box {
            background: #1e293b; border: 1px solid #38bdf8; border-radius: 12px;
            padding: 14px; text-align: center; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
        }
        .now-playing-title { font-weight: bold; font-size: 18px; color: #38bdf8; margin-top: 4px; }
        .lyrics-box { color: #f59e0b; font-size: 15px; font-weight: bold; min-height: 24px; margin-top: 8px; font-style: italic; }
        .controls-row { display: flex; gap: 8px; justify-content: center; margin-top: 12px; }
        .btn-ctrl {
            flex: 1; padding: 10px 12px; font-size: 14px; font-weight: bold;
            border-radius: 8px; border: none; color: white; cursor: pointer;
        }
        .btn-play { background: #16a34a; }
        .btn-skip { background: #d97706; }
        input { width: 100%; padding: 12px; box-sizing: border-box; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: white; font-size: 16px; margin-bottom: 15px; }
        ul { list-style: none; padding: 0; margin: 0; }
        li { background: #1e293b; margin-bottom: 8px; padding: 12px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #334155; }
        .btn-group { display: flex; gap: 6px; }
        button.btn-add { background: #2563eb; color: white; border: none; padding: 8px 12px; border-radius: 6px; font-weight: bold; }
        button.btn-insert { background: #d97706; color: white; border: none; padding: 8px 12px; border-radius: 6px; font-weight: bold; }
        .queue-item { background: #0f172a; border-left: 4px solid #38bdf8; }
        .suggestion-item { background: #1e293b; border-left: 4px solid #f59e0b; }
        .progress-bar-container { background: #334155; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 10px; }
        .progress-bar-fill { background: #38bdf8; height: 100%; width: 0%; transition: width 0.3s ease; }
    </style>
</head>
<body>
    <h2>🎤 KTV Remote</h2>

    <div class="now-playing-box">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Now Playing</div>
        <div id="nowPlayingText" class="now-playing-title">None</div>
        <div id="lyricsText" class="lyrics-box"></div>
        <div class="progress-bar-container"><div id="progressFill" class="progress-bar-fill"></div></div>

        <div class="controls-row">
            <button id="btnPlayPause" class="btn-ctrl btn-play" onclick="triggerAction('play_pause')">⏯ Pause</button>
            <button class="btn-ctrl btn-skip" onclick="triggerAction('skip')">⏭ Skip</button>
        </div>
    </div>

    <h3>Current Queue</h3>
    <ul id="queueList"></ul>

    <hr style="border-color: #334155; margin: 20px 0;">
    <h3>💡 Suggested Songs</h3>
    <ul id="suggestionsList"></ul>

    <hr style="border-color: #334155; margin: 20px 0;">
    <h3>Song Library</h3>
    <input type="text" id="searchInput" onkeyup="filterSongs()" placeholder="Search library...">
    <ul id="libraryList"></ul>

    <script>
        const socket = io();
        let fullLibrary = [];

        socket.on('state_update', data => {
            document.getElementById('nowPlayingText').innerText = data.now_playing || 'None';
            document.getElementById('btnPlayPause').innerText = data.is_playing ? '⏸ Pause' : '▶ Play';

            const queueList = document.getElementById('queueList');
            if (!data.queue || data.queue.length === 0) {
                queueList.innerHTML = '<li style="color:#64748b;">Queue is empty</li>';
            } else {
                queueList.innerHTML = data.queue.map((s, i) => `<li class="queue-item"><span>${i+1}. ${s}</span></li>`).join('');
            }
        });

        socket.on('telemetry', data => {
            if (data.length_ms > 0) {
                const pct = (data.time_ms / data.length_ms) * 100;
                document.getElementById('progressFill').style.width = pct + '%';
            }
            if (data.lyric) {
                document.getElementById('lyricsText').innerText = data.lyric;
            }
        });

        function fetchStaticData() {
            fetch('/api/library').then(r => r.json()).then(data => {
                fullLibrary = data.library;
                renderLibrary(fullLibrary);
            });
            fetch('/api/suggestions').then(r => r.json()).then(data => {
                const list = document.getElementById('suggestionsList');
                list.innerHTML = data.suggestions.map(song => `
                    <li class="suggestion-item">
                        <span>⭐ ${song}</span>
                        <div class="btn-group">
                            <button class="btn-add" onclick="addSong('${song.replace(/'/g, "\\'")}', false)">Add</button>
                            <button class="btn-insert" onclick="addSong('${song.replace(/'/g, "\\'")}', true)">⚡ Insert</button>
                        </div>
                    </li>
                `).join('');
            });
        }

        function renderLibrary(songs) {
            const list = document.getElementById('libraryList');
            list.innerHTML = songs.map(song => `
                <li>
                    <span>${song}</span>
                    <div class="btn-group">
                        <button class="btn-add" onclick="addSong('${song.replace(/'/g, "\\'")}', false)">Add</button>
                        <button class="btn-insert" onclick="addSong('${song.replace(/'/g, "\\'")}', true)">⚡ Insert</button>
                    </div>
                </li>
            `).join('');
        }

        function filterSongs() {
            const query = document.getElementById('searchInput').value.toLowerCase();
            renderLibrary(fullLibrary.filter(s => s.toLowerCase().includes(query)));
        }

        function addSong(songTitle, insertNext) {
            fetch('/api/queue', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ song: songTitle, insert_next: insertNext })
            });
        }

        function triggerAction(actionName) {
            fetch('/api/action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ action: actionName })
            });
        }

        fetchStaticData();
    </script>
</body>
</html>
"""


def broadcast_system_state():
    if control_win:
        now_playing = (
            control_win.current_song["name"]
            if control_win.current_song
            else "None"
        )
        is_playing = control_win.media_player.is_playing() == 1
        queue_titles = [s["name"] for s in control_win.selected_queue]

        socketio.emit(
            "state_update",
            {
                "now_playing": now_playing,
                "is_playing": is_playing,
                "queue": queue_titles,
            },
        )


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/library", methods=["GET"])
def get_library():
    if control_win:
        titles = [s["name"] for s in control_win.song_library]
        return jsonify({"library": sorted(titles)})
    return jsonify({"library": []})


@app.route("/api/suggestions", methods=["GET"])
def get_suggestions():
    if control_win and control_win.song_library:
        sample_size = min(len(control_win.song_library), 5)
        sampled_songs = random.sample(control_win.song_library, sample_size)
        return jsonify({"suggestions": [s["name"] for s in sampled_songs]})
    return jsonify({"suggestions": []})


@app.route("/api/queue", methods=["POST"])
def post_queue():
    data = request.get_json()
    song_name = data.get("song")
    insert_next = data.get("insert_next", False)

    if song_name and control_win:
        matched_song = next(
            (s for s in control_win.song_library if s["name"] == song_name),
            None,
        )
        if matched_song:
            control_win.web_song_added.emit(matched_song, insert_next)
            return jsonify({"status": "success", "song": song_name}), 200

    return jsonify({"status": "error", "message": "Song not found"}), 400


@app.route("/api/action", methods=["POST"])
def trigger_action():
    data = request.get_json()
    action = data.get("action")

    if action and control_win:
        control_win.web_action_triggered.emit(action)
        return jsonify({"status": "success", "action": action}), 200

    return jsonify({"status": "error", "message": "Invalid action"}), 400


@socketio.on("connect")
def handle_connect():
    broadcast_system_state()


def start_flask_server():
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


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