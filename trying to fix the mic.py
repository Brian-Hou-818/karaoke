import os

# Force Qt to use X11/XCB on Linux for proper VLC video embedding
os.environ["QT_QPA_PLATFORM"] = "xcb"

import random
import sys
import threading
from pathlib import Path

import numpy as np
import pyaudio

from flask import Flask, jsonify, render_template_string, request
from flask_socketio import SocketIO, emit
import vlc

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
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
# MICROPHONE / SINGING AUDIO ENGINE (CRASH-PROOF CIRCULAR BUFFER)
# ==============================================================================
class MicAudioEngine:
    """Handles real-time mic processing with a safe, circular ring buffer."""

    def __init__(self, sample_rate=44100, chunk_size=512):
        self.p = pyaudio.PyAudio()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        self.is_running = False

        self.mic_volume = 1.0
        self.input_device_index = None
        self.output_device_index = None

        self.reverb_enabled = False
        self.delay_ms = 150
        self.reverb_feedback = 0.4

        self.lock = threading.Lock()

        # Fixed 2-second circular buffer to handle dynamic delay adjustments cleanly
        self.buffer_capacity = self.sample_rate * 2
        self.delay_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.write_pos = 0
        self.delay_samples = int(self.sample_rate * (self.delay_ms / 1000.0))

    def set_delay_ms(self, delay_ms):
        with self.lock:
            self.delay_ms = max(20, min(delay_ms, 500))
            self.delay_samples = int(self.sample_rate * (self.delay_ms / 1000.0))

    def set_reverb_feedback(self, feedback_pct):
        with self.lock:
            self.reverb_feedback = (feedback_pct / 100.0) * 0.85

    def set_reverb(self, enabled):
        with self.lock:
            self.reverb_enabled = enabled

    def get_input_devices(self):
        devices = []
        for i in range(self.p.get_device_count()):
            try:
                dev = self.p.get_device_info_by_index(i)
                if dev.get("maxInputChannels", 0) > 0:
                    devices.append((i, dev.get("name", f"Input Device {i}")))
            except Exception:
                pass
        return devices

    def get_output_devices(self):
        devices = []
        for i in range(self.p.get_device_count()):
            try:
                dev = self.p.get_device_info_by_index(i)
                if dev.get("maxOutputChannels", 0) > 0:
                    devices.append((i, dev.get("name", f"Output Device {i}")))
            except Exception:
                pass
        return devices

    def set_input_device(self, device_index):
        self.input_device_index = device_index
        if self.is_running:
            self.restart_stream()

    def set_output_device(self, device_index):
        self.output_device_index = device_index
        if self.is_running:
            self.restart_stream()

    def start_mic(self):
        if self.is_running:
            return
        self.is_running = True
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                output=True,
                input_device_index=self.input_device_index,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"[Mic Error]: Failed to open audio device stream: {e}")
            self.is_running = False

    def stop_mic(self):
        if self.stream:
            self.is_running = False
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def restart_stream(self):
        self.stop_mic()
        self.start_mic()

    def set_volume(self, val_pct):
        with self.lock:
            self.mic_volume = val_pct / 100.0

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Thread-safe real-time callback with fixed circular buffer indexing."""
        if not self.is_running or in_data is None:
            return (None, pyaudio.paComplete)

        try:
            input_signal = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            chunk_len = len(input_signal)

            with self.lock:
                vol = self.mic_volume
                reverb_on = self.reverb_enabled
                fb = self.reverb_feedback
                delay_smp = self.delay_samples

                dry_signal = input_signal * vol

                if reverb_on:
                    read_pos = (self.write_pos - delay_smp) % self.buffer_capacity
                    indices = (np.arange(chunk_len) + read_pos) % self.buffer_capacity
                    echo_signal = self.delay_buffer[indices] * fb

                    output_signal = dry_signal + echo_signal

                    write_indices = (np.arange(chunk_len) + self.write_pos) % self.buffer_capacity
                    self.delay_buffer[write_indices] = output_signal
                    self.write_pos = (self.write_pos + chunk_len) % self.buffer_capacity
                else:
                    output_signal = dry_signal

            output_signal = np.clip(output_signal, -32768, 32767)
            out_bytes = output_signal.astype(np.int16).tobytes()

            return (out_bytes, pyaudio.paContinue)

        except Exception:
            return (in_data, pyaudio.paContinue)

    def terminate(self):
        self.stop_mic()
        self.p.terminate()


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
        self.setWindowTitle("MP4 Video Display")
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
# MAIN MP4 PLAYER CONTROL WINDOW
# ==============================================================================
class MP4PlayerControlWindow(QMainWindow):
    web_song_added = pyqtSignal(dict, bool)
    web_action_triggered = pyqtSignal(str)

    def __init__(self, display_window):
        super().__init__()
        self.display_win = display_window
        self.display_win.skip_callback = self.play_next

        self.setWindowTitle("MP4 Video Player Control Panel")
        self.setGeometry(80, 80, 980, 840)

        self.current_folder = str(Path.home() / "Downloads")

        self.song_library = []
        self.selected_queue = []
        self.current_song = None

        self.playback_rate = 1.0

        self.mic_engine = MicAudioEngine()

        vlc_flags = [
            "--aout=pulse",
            "--role=music",
            "--avcodec-hw=none",
            "--no-video-title-show",
            "--quiet",
        ]
        self.vlc_instance = vlc.Instance(" ".join(vlc_flags))
        if self.vlc_instance is None:
            raise RuntimeError("Failed to initialize libVLC instance.")

        self.media_player = self.vlc_instance.media_player_new()

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

        self.scan_folder_for_songs(self.current_folder)

    def closeEvent(self, event):
        self.mic_engine.terminate()
        super().closeEvent(event)

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
            QLineEdit, QComboBox { 
                background-color: #0f172a; color: white; 
                border: 1px solid #334155; border-radius: 6px; padding: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: white;
                selection-background-color: #2563eb;
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

        music_box = QGroupBox("🎬 Playback Controls")
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

        self.btn_skip = QPushButton("⏭ Skip Video (N)")
        self.btn_skip.clicked.connect(self.play_next)
        playback_row.addWidget(self.btn_skip)
        music_layout.addLayout(playback_row)

        tempo_row = QHBoxLayout()
        tempo_row.addWidget(QLabel("<b>Playback Speed:</b>"))
        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setRange(80, 120)
        self.tempo_slider.setValue(100)
        self.tempo_slider.valueChanged.connect(self.change_playback_speed)
        tempo_row.addWidget(self.tempo_slider, stretch=1)
        self.tempo_label = QLabel("1.0x")
        tempo_row.addWidget(self.tempo_label)
        music_layout.addLayout(tempo_row)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("<b>Volume:</b>"))
        self.music_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.music_vol_slider.setRange(0, 100)
        self.music_vol_slider.setValue(80)
        self.music_vol_slider.valueChanged.connect(self.change_music_volume)
        vol_row.addWidget(self.music_vol_slider, stretch=1)
        self.music_vol_label = QLabel("80%")
        vol_row.addWidget(self.music_vol_label)
        music_layout.addLayout(vol_row)

        left_col.addWidget(music_box)

        singing_box = QGroupBox("🎤 Karaoke & Audio Effects")
        singing_layout = QVBoxLayout(singing_box)

        singing_layout.addWidget(QLabel("<b>Input Device (Mic):</b>"))
        self.input_combo = QComboBox()
        self.populate_input_devices()
        self.input_combo.currentIndexChanged.connect(
            self.on_input_device_changed
        )
        singing_layout.addWidget(self.input_combo)

        singing_layout.addWidget(
            QLabel("<b>Output Device (Speakers/Mics):</b>")
        )
        self.output_combo = QComboBox()
        self.populate_output_devices()
        self.output_combo.currentIndexChanged.connect(
            self.on_output_device_changed
        )
        singing_layout.addWidget(self.output_combo)

        mic_ctrl_row = QHBoxLayout()
        self.btn_toggle_mic = QPushButton("🎙️ Mic OFF")
        self.btn_toggle_mic.clicked.connect(self.toggle_microphone)
        mic_ctrl_row.addWidget(self.btn_toggle_mic)

        self.chk_reverb = QCheckBox("✨ Enable Echo/Reverb")
        self.chk_reverb.toggled.connect(self.mic_engine.set_reverb)
        mic_ctrl_row.addWidget(self.chk_reverb)
        singing_layout.addLayout(mic_ctrl_row)

        mic_vol_row = QHBoxLayout()
        mic_vol_row.addWidget(QLabel("<b>Mic Gain (Up to 10x):</b>"))
        self.mic_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic_vol_slider.setRange(0, 1000)
        self.mic_vol_slider.setValue(100)
        self.mic_vol_slider.valueChanged.connect(self.change_mic_volume)
        mic_vol_row.addWidget(self.mic_vol_slider, stretch=1)
        self.mic_vol_label = QLabel("1.0x (100%)")
        mic_vol_row.addWidget(self.mic_vol_label)
        singing_layout.addLayout(mic_vol_row)

        delay_row = QHBoxLayout()
        delay_row.addWidget(QLabel("<b>Echo Delay:</b>"))
        self.delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.delay_slider.setRange(20, 500)
        self.delay_slider.setValue(150)
        self.delay_slider.valueChanged.connect(self.change_delay_ms)
        delay_row.addWidget(self.delay_slider, stretch=1)
        self.delay_label = QLabel("150 ms")
        delay_row.addWidget(self.delay_label)
        singing_layout.addLayout(delay_row)

        reverb_row = QHBoxLayout()
        reverb_row.addWidget(QLabel("<b>Reverb Decay:</b>"))
        self.reverb_slider = QSlider(Qt.Orientation.Horizontal)
        self.reverb_slider.setRange(0, 90)
        self.reverb_slider.setValue(40)
        self.reverb_slider.valueChanged.connect(self.change_reverb_feedback)
        reverb_row.addWidget(self.reverb_slider, stretch=1)
        self.reverb_label = QLabel("40%")
        reverb_row.addWidget(self.reverb_label)
        singing_layout.addLayout(reverb_row)

        left_col.addWidget(singing_box)

        grid_layout.addLayout(left_col, stretch=1)

        right_col = QVBoxLayout()
        self.tab_widget = QTabWidget()

        browse_tab = QWidget()
        browse_layout = QVBoxLayout(browse_tab)
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search video title...")
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

        self.btn_insert_browser = QPushButton("⚡ Insert Next")
        self.btn_insert_browser.setObjectName("priorityBtn")
        self.btn_insert_browser.clicked.connect(
            lambda: self.add_selected_song_from_browser(insert_next=True)
        )
        browse_btn_row.addWidget(self.btn_insert_browser)
        browse_layout.addLayout(browse_btn_row)

        self.tab_widget.addTab(browse_tab, "🔍 Browse Videos")

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

        self.btn_insert_suggestion = QPushButton("⚡ Insert Next")
        self.btn_insert_suggestion.setObjectName("priorityBtn")
        self.btn_insert_suggestion.clicked.connect(
            lambda: self.add_selected_song_from_suggestions(insert_next=True)
        )
        sug_btn_row.addWidget(self.btn_insert_suggestion)
        suggestions_layout.addLayout(sug_btn_row)

        self.tab_widget.addTab(suggestions_tab, "💡 Suggestions")
        right_col.addWidget(self.tab_widget, stretch=1)

        right_col.addWidget(
            QLabel("<b>📋 Playback Queue</b> <i>(Drag to reorder)</i>")
        )
        self.queue_widget = DraggableQueueList()
        self.queue_widget.reorder_callback = self.on_queue_reordered
        right_col.addWidget(self.queue_widget, stretch=1)

        btn_remove = QPushButton("❌ Remove Selected")
        btn_remove.clicked.connect(self.remove_from_queue)
        right_col.addWidget(btn_remove)

        grid_layout.addLayout(right_col, stretch=1)
        root_layout.addLayout(grid_layout)

    def populate_input_devices(self):
        self.input_combo.clear()
        self.input_combo.addItem("Default Microphone", None)
        devices = self.mic_engine.get_input_devices()
        for idx, name in devices:
            self.input_combo.addItem(f"{name} (ID: {idx})", idx)

    def populate_output_devices(self):
        self.output_combo.clear()
        self.output_combo.addItem("Default Output Speaker", None)
        devices = self.mic_engine.get_output_devices()
        for idx, name in devices:
            self.output_combo.addItem(f"{name} (ID: {idx})", idx)

    def on_input_device_changed(self, index):
        dev_idx = self.input_combo.itemData(index)
        self.mic_engine.set_input_device(dev_idx)

    def on_output_device_changed(self, index):
        dev_idx = self.output_combo.itemData(index)
        self.mic_engine.set_output_device(dev_idx)

    def toggle_microphone(self):
        if self.mic_engine.is_running:
            self.mic_engine.stop_mic()
            self.btn_toggle_mic.setText("🎙️ Mic OFF")
            self.btn_toggle_mic.setStyleSheet("")
        else:
            self.mic_engine.start_mic()
            self.btn_toggle_mic.setText("🔴 Mic ON")
            self.btn_toggle_mic.setStyleSheet("background-color: #dc2626;")

    def change_mic_volume(self, value):
        self.mic_engine.set_volume(value)
        self.mic_vol_label.setText(f"{value/100.0:.1f}x ({value}%)")

    def change_delay_ms(self, value):
        self.mic_engine.set_delay_ms(value)
        self.delay_label.setText(f"{value} ms")

    def change_reverb_feedback(self, value):
        self.mic_engine.set_reverb_feedback(value)
        self.reverb_label.setText(f"{value}%")

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
            self, "Select Video Directory", self.current_folder
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

            socketio.emit(
                "telemetry",
                {
                    "time_ms": curr_time_ms,
                    "length_ms": total_time_ms,
                },
            )


# ==============================================================================
# FLASK WEB SERVER + SOCKET.IO WEBSOCKET MOBILE REMOTE
# ==============================================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = "mp4_player_secret_key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
control_win = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MP4 Player Mobile Remote</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 15px; background: #0f172a; color: #fff; margin: 0; }
        h2, h3 { text-align: center; margin-top: 10px; color: #38bdf8; }
        .now-playing-box {
            background: #1e293b; border: 1px solid #38bdf8; border-radius: 12px;
            padding: 14px; text-align: center; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
        }
        .now-playing-title { font-weight: bold; font-size: 18px; color: #38bdf8; margin-top: 4px; }
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
    <h2>📱 MP4 Player Remote</h2>

    <div class="now-playing-box">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Now Playing</div>
        <div id="nowPlayingText" class="now-playing-title">None</div>
        <div class="progress-bar-container"><div id="progressFill" class="progress-bar-fill"></div></div>

        <div class="controls-row">
            <button id="btnPlayPause" class="btn-ctrl btn-play" onclick="triggerAction('play_pause')">⏯ Pause</button>
            <button class="btn-ctrl btn-skip" onclick="triggerAction('skip')">⏭ Skip</button>
        </div>
    </div>

    <h3>Current Queue</h3>
    <ul id="queueList"></ul>

    <hr style="border-color: #334155; margin: 20px 0;">
    <h3>💡 Suggested Videos</h3>
    <ul id="suggestionsList"></ul>

    <hr style="border-color: #334155; margin: 20px 0;">
    <h3>Video Library</h3>
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
    control_win = MP4PlayerControlWindow(display_win)

    server_thread = threading.Thread(target=start_flask_server, daemon=True)
    server_thread.start()

    display_win.show()
    control_win.show()

    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()