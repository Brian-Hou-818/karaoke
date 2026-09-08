import os
import sys
import ctypes
import random
from pathlib import Path
import sounddevice as sd
import numpy as np

# SciPy DSP for Parametric EQ calculation
from scipy.signal import butter, iirnotch, sosfilt

# ==============================================================================
# 1. WINDOWS / LINUX DLL PATCH & SETUP FOR PYTHON-VLC
# ==============================================================================
VLC_PATH = r"C:\Program Files\VideoLAN\VLC"

if os.path.exists(VLC_PATH):
    os.add_dll_directory(VLC_PATH)
    os.environ['PATH'] = VLC_PATH + os.pathsep + os.environ['PATH']
    os.environ['PYTHON_VLC_MODULE_PATH'] = VLC_PATH

_orig_cdll_init = ctypes.CDLL.__init__


def _patched_cdll_init(self, name, *args, **kwargs):
    if isinstance(name, str) and ("libvlc" in name or name.startswith(".\\")):
        dll_name = os.path.basename(name)
        abs_vlc_path = os.path.join(VLC_PATH, dll_name)
        if os.path.exists(abs_vlc_path):
            name = abs_vlc_path
        kwargs['winmode'] = 0
    _orig_cdll_init(self, name, *args, **kwargs)


ctypes.CDLL.__init__ = _patched_cdll_init

import vlc

ctypes.CDLL.__init__ = _orig_cdll_init

# ==============================================================================
# 2. APPLICATION IMPORTS
# ==============================================================================
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QLabel, QFrame, QLineEdit,
    QListWidgetItem, QSlider, QComboBox, QAbstractItemView, QCheckBox,
    QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent, QShortcut, QKeySequence


# Helper function to truncate strings to a maximum character length for "Now Playing"
def get_truncated_title(title, max_length=10):
    if len(title) > max_length:
        return title[: max_length - 3] + "..."
    return title


# ==============================================================================
# 3. DSP PARAMETRIC EQUALIZER FUNCTION (VOCAL BAND SUPPRESSION)
# ==============================================================================
def apply_parametric_vocal_eq(audio_data, sample_rate=44100):
    """
    Applies a Parametric EQ filter targeting human voice frequencies:
    - High-Pass Filter @ 100 Hz (cuts low vocal rumble)
    - Low-Pass Filter @ 8,000 Hz (cuts high vocal sibilance 'S'/'T' sounds)
    - Wide Q Notch Filters scoops 80 Hz - 4,000 Hz vocal formants (-18 dB attenuation)
    """
    if audio_data.size == 0:
        return audio_data

    processed = audio_data.copy()

    # 1. High-Pass Filter (< 100 Hz cut)
    hp_sos = butter(2, 100, btype='highpass', fs=sample_rate, output='sos')
    processed = sosfilt(hp_sos, processed, axis=0)

    # 2. Low-Pass Filter (> 8,000 Hz cut)
    lp_sos = butter(2, 8000, btype='lowpass', fs=sample_rate, output='sos')
    processed = sosfilt(lp_sos, processed, axis=0)

    # 3. Parametric Vocal Notch Band Cuts (80 Hz - 4,000 Hz core vocal range)
    vocal_center_frequencies = [250, 800, 1500, 3000]
    wide_q_factor = 0.85  # Covers broad vocal band

    for freq in vocal_center_frequencies:
        b, a = iirnotch(freq, wide_q_factor, fs=sample_rate)
        if processed.ndim > 1:
            for ch in range(processed.shape[1]):
                processed[:, ch] = np.convolve(processed[:, ch], b, mode='same')
        else:
            processed = np.convolve(processed, b, mode='same')

    return np.clip(processed * 0.3, -1.0, 1.0)


# ==============================================================================
# 4. LOW-LATENCY SOFTWARE AUDIO STREAM WITH ECHO / REVERB DSP
# ==============================================================================
class AudioPassthroughStream:
    """ Ultra-low latency audio stream with optimized feedback echo/reverb DSP """

    def __init__(self, input_device_id, output_device_id=None, sample_rate=44100):
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.sample_rate = sample_rate
        self.volume = 1.0

        # DSP Echo Parameters
        self.echo_delay_ms = 180
        self.echo_feedback = 0.4

        # Pre-allocated delay buffer
        self.buffer_size = sample_rate * 2
        self.delay_buffer = np.zeros((self.buffer_size, 2), dtype=np.float32)
        self.write_pos = 0

        self.is_running = False
        self.stream = None

    def _audio_callback(self, indata, outdata, frames, time, status):
        out_channels = outdata.shape[1]

        # Quick channel matching without unnecessary copies
        if indata.shape[1] == 1 and out_channels == 2:
            in_samples = np.column_stack((indata[:, 0], indata[:, 0]))
        else:
            in_samples = indata[:, :out_channels]

        # Apply Mic Gain
        processed = in_samples * self.volume

        # Vectorized ring-buffer read/write for low-latency processing
        delay_samples = int((self.echo_delay_ms / 1000.0) * self.sample_rate)
        read_indices = (np.arange(self.write_pos, self.write_pos + frames) - delay_samples) % self.buffer_size
        write_indices = (np.arange(self.write_pos, self.write_pos + frames)) % self.buffer_size

        delayed_samples = self.delay_buffer[read_indices, :out_channels]
        mixed_samples = processed + (delayed_samples * self.echo_feedback)

        # Write mixed samples back to buffer for feedback decay
        self.delay_buffer[write_indices, :out_channels] = mixed_samples
        self.write_pos = (self.write_pos + frames) % self.buffer_size

        outdata[:] = np.clip(mixed_samples, -1.0, 1.0)

    def start(self):
        if self.is_running:
            return
        try:
            dev_info = sd.query_devices(self.input_device_id)
            channels = min(dev_info['max_input_channels'], 2)

            device_tuple = (self.input_device_id, self.output_device_id)

            # --- LOW-LATENCY OPTIMIZATIONS ---
            self.stream = sd.Stream(
                device=device_tuple,
                samplerate=self.sample_rate,
                blocksize=64,
                latency='low',
                channels=channels,
                dtype='float32',
                callback=self._audio_callback
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
                    latency='low',
                    channels=channels,
                    dtype='float32',
                    callback=self._audio_callback
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
    """ Custom QListWidget that handles drag-and-drop reordering """

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
    """ Standalone window dedicated purely to video output """

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
        elif event.key() in (Qt.Key.Key_N, Qt.Key.Key_Right, Qt.Key.Key_MediaNext):
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
    def __init__(self, display_window):
        super().__init__()
        self.display_win = display_window
        self.display_win.skip_callback = self.play_next

        self.setWindowTitle("KTV Control Panel")
        self.setGeometry(80, 80, 950, 950)

        # Default Library Directory to Downloads folder
        self.current_folder = str(Path.home() / "Downloads")

        # App State
        self.song_library = []
        self.selected_queue = []
        self.current_song = None
        self.vocal_eq_active = False

        # Audio Streams
        self.mic1_stream = None
        self.mic2_stream = None

        # VLC Engine
        self.vlc_instance = vlc.Instance('--aout=directsound' if sys.platform.startswith('win') else '')
        self.media_player = self.vlc_instance.media_player_new()

        # Polling Timer for Autoplay / Media End Detection
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(500)
        self.poll_timer.timeout.connect(self.check_media_status)
        self.poll_timer.start()

        self.init_ui()
        self.setup_shortcuts()
        self.populate_audio_devices()

        # Initial scan of Downloads directory
        self.scan_folder_for_songs(self.current_folder)

    def setup_shortcuts(self):
        """ Configures hotkeys to skip songs from the control window """
        self.shortcut_n = QShortcut(QKeySequence("N"), self)
        self.shortcut_n.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_n.activated.connect(self.play_next)

        self.shortcut_ctrl_right = QShortcut(QKeySequence("Ctrl+Right"), self)
        self.shortcut_ctrl_right.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_ctrl_right.activated.connect(self.play_next)

        self.shortcut_media_next = QShortcut(QKeySequence(Qt.Key.Key_MediaNext), self)
        self.shortcut_media_next.setContext(Qt.ShortcutContext.ApplicationShortcut)
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

            /* LARGE SPACIOUS SLIDERS */
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

        # ==============================================================================
        # TOP TOOLBAR: STATUS & GLOBAL CONTROLS
        # ==============================================================================
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

        # ==============================================================================
        # MAIN GRID CONTENT
        # ==============================================================================
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(16)

        # ------------------------------------------------------------------------------
        # LEFT COLUMN: AUDIO, MUSIC & MIC CONTROL CARDS
        # ------------------------------------------------------------------------------
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # 0. HARDWARE AUDIO OUTPUT SELECTION CARD
        out_box = QGroupBox("🔊 Output Hardware Device")
        out_layout = QVBoxLayout(out_box)
        out_layout.setSpacing(10)

        out_dev_row = QHBoxLayout()
        out_dev_row.addWidget(QLabel("<b>Output:</b>"))
        self.output_combo = QComboBox()
        self.output_combo.currentIndexChanged.connect(self.on_output_device_changed)
        out_dev_row.addWidget(self.output_combo, stretch=1)
        out_layout.addLayout(out_dev_row)

        left_col.addWidget(out_box)

        # 1. MUSIC PLAYBACK CARD
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
        self.btn_stereo.clicked.connect(lambda: self.set_audio_channel("stereo"))
        track_row.addWidget(self.btn_stereo)

        self.btn_left = QPushButton("Music (L)")
        self.btn_left.clicked.connect(lambda: self.set_audio_channel("left"))
        track_row.addWidget(self.btn_left)

        self.btn_right = QPushButton("Vocal (R)")
        self.btn_right.clicked.connect(lambda: self.set_audio_channel("right"))
        track_row.addWidget(self.btn_right)
        music_layout.addLayout(track_row)

        # PARAMETRIC VOCAL EQ DSP TOGGLE
        eq_row = QHBoxLayout()
        self.btn_vocal_eq = QPushButton("🎙️ Vocal Parametric EQ (80-4kHz Cut): OFF")
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

        # 2. MICROPHONE 1 CARD
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

        # 3. MICROPHONE 2 CARD
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

        # 4. MASTER ECHO & REVERB DSP CARD
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

        # ------------------------------------------------------------------------------
        # RIGHT COLUMN: SEARCH BROWSER & DRAG/DROP QUEUE
        # ------------------------------------------------------------------------------
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # 1. SEARCH BROWSER
        right_col.addWidget(QLabel("<b>🔍 Browse Song Library</b>"))
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search title or artist...")
        self.search_bar.textChanged.connect(self.filter_songs)
        right_col.addWidget(self.search_bar)

        self.library_list_widget = QListWidget()
        self.library_list_widget.itemDoubleClicked.connect(self.add_selected_song_from_browser)
        right_col.addWidget(self.library_list_widget, stretch=1)

        self.btn_add_browser = QPushButton("➕ Add Selected to Queue")
        self.btn_add_browser.setObjectName("primaryBtn")
        self.btn_add_browser.clicked.connect(self.add_selected_song_from_browser)
        right_col.addWidget(self.btn_add_browser)

        # 2. SELECTED SONGS QUEUE
        right_col.addWidget(QLabel("<b>📋 Selected Songs Queue</b> <i>(Drag to reorder)</i>"))

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
        """ Populate input (mics) and output (speakers) hardware devices """
        self.mic1_combo.clear()
        self.mic2_combo.clear()
        self.output_combo.clear()

        devices = sd.query_devices()
        default_out_idx = sd.default.device[1]

        for idx, dev in enumerate(devices):
            # Input devices
            if dev['max_input_channels'] > 0:
                name = f"[{idx}] {dev['name']}"
                self.mic1_combo.addItem(name, userData=idx)
                self.mic2_combo.addItem(name, userData=idx)

            # Output devices
            if dev['max_output_channels'] > 0:
                name = f"[{idx}] {dev['name']}"
                self.output_combo.addItem(name, userData=idx)

        if self.mic2_combo.count() > 1:
            self.mic2_combo.setCurrentIndex(1)

        # Select default system output device in combo box
        for i in range(self.output_combo.count()):
            if self.output_combo.itemData(i) == default_out_idx:
                self.output_combo.setCurrentIndex(i)
                break

    def get_selected_output_device_id(self):
        return self.output_combo.currentData()

    def on_output_device_changed(self):
        out_id = self.get_selected_output_device_id()

        # Update Mic 1
        if self.mic1_stream and self.mic1_stream.is_running:
            self.mic1_stream.stop()
            self.mic1_stream = AudioPassthroughStream(self.mic1_stream.input_device_id, out_id)
            self.mic1_stream.start()

        # Update Mic 2
        if self.mic2_stream and self.mic2_stream.is_running:
            self.mic2_stream.stop()
            self.mic2_stream = AudioPassthroughStream(self.mic2_stream.input_device_id, out_id)
            self.mic2_stream.start()

        self.update_mic_settings()
        self.sync_vlc_output_device()

    def sync_vlc_output_device(self):
        """ Syncs VLC audio output with the selected sounddevice output """
        out_id = self.get_selected_output_device_id()
        if out_id is None:
            return

        target_name = sd.query_devices(out_id)['name']

        # Enumerate VLC audio outputs to find matching device
        device_enum = self.media_player.audio_output_device_enum()
        if device_enum:
            curr = device_enum
            while curr:
                dev_id = curr.contents.device
                dev_desc = curr.contents.description.decode('utf-8', errors='ignore') if curr.contents.description else ""
                if target_name.lower() in dev_desc.lower() or dev_desc.lower() in target_name.lower():
                    self.media_player.audio_output_device_set(None, dev_id)
                    break
                curr = curr.contents.next

            # FIX: Correct method name in python-vlc for releasing device list
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
                if file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                    full_path = os.path.join(root, file)
                    song_name = os.path.splitext(file)[0]
                    self.song_library.append({'name': song_name, 'path': full_path})

        self.populate_library_list(self.song_library)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select KTV Songs Folder", self.current_folder)
        if folder_path:
            self.scan_folder_for_songs(folder_path)

    def refresh_library(self):
        self.scan_folder_for_songs(self.current_folder)
        if self.search_bar.text():
            self.filter_songs(self.search_bar.text())

    def populate_library_list(self, songs):
        self.library_list_widget.clear()
        for song in songs:
            item = QListWidgetItem(song['name'])
            item.setData(Qt.ItemDataRole.UserRole, song)
            self.library_list_widget.addItem(item)

    def filter_songs(self, text):
        search_term = text.lower().strip()
        if not search_term:
            self.populate_library_list(self.song_library)
            return

        filtered = [s for s in self.song_library if search_term in s['name'].lower()]
        self.populate_library_list(filtered)

    def add_selected_song_from_browser(self):
        current_item = self.library_list_widget.currentItem()
        if current_item:
            song = current_item.data(Qt.ItemDataRole.UserRole)
            self.add_song_to_queue(song)

    def add_song_to_queue(self, song):
        self.selected_queue.append(song)
        self.refresh_queue_widget()

        if not self.media_player.is_playing() and not self.current_song:
            self.play_next()

    def remove_from_queue(self):
        selected_row = self.queue_widget.currentRow()
        if 0 <= selected_row < len(self.selected_queue):
            del self.selected_queue[selected_row]
            self.refresh_queue_widget()

    def on_queue_reordered(self):
        new_queue = []
        for i in range(self.queue_widget.count()):
            item = self.queue_widget.item(i)
            song_data = item.data(Qt.ItemDataRole.UserRole)
            if song_data:
                new_queue.append(song_data)
        self.selected_queue = new_queue
        self.refresh_queue_widget()

    def refresh_queue_widget(self):
        self.queue_widget.blockSignals(True)
        self.queue_widget.clear()
        for idx, song in enumerate(self.selected_queue, start=1):
            item = QListWidgetItem(f"≡  {idx}. {song['name']}")
            item.setData(Qt.ItemDataRole.UserRole, song)
            self.queue_widget.addItem(item)
        self.queue_widget.blockSignals(False)

    def play_next(self):
        if self.selected_queue:
            self.current_song = self.selected_queue.pop(0)
            self.refresh_queue_widget()
            display_name = get_truncated_title(self.current_song['name'], 10)
            self.now_playing_label.setText(f"<b>Now Playing:</b> {display_name}")
        elif self.chk_auto_random.isChecked() and self.song_library:
            self.current_song = random.choice(self.song_library)
            display_name = get_truncated_title(self.current_song['name'], 10)
            self.now_playing_label.setText(f"<b>Now Playing (Random):</b> 🎲 {display_name}")
        else:
            self.current_song = None
            self.media_player.stop()
            self.now_playing_label.setText("<b>Now Playing:</b> Finished")
            self.btn_play_pause.setText("▶ Play")
            return

        media = self.vlc_instance.media_new(self.current_song['path'])
        self.media_player.set_media(media)

        # Cross-platform window embedding
        if sys.platform.startswith('win'):
            self.media_player.set_hwnd(int(self.display_win.video_frame.winId()))
        else:
            self.media_player.set_xwindow(int(self.display_win.video_frame.winId()))

        self.media_player.play()
        self.sync_vlc_output_device()
        self.media_player.audio_set_volume(self.music_vol_slider.value())
        self.btn_play_pause.setText("⏸ Pause")

        if self.vocal_eq_active:
            self.apply_vlc_vocal_eq(True)

    def check_media_status(self):
        state = self.media_player.get_state()
        if state == vlc.State.Ended:
            if self.chk_autoplay.isChecked():
                self.play_next()
            else:
                self.current_song = None
                self.now_playing_label.setText("<b>Now Playing:</b> Finished")
                self.btn_play_pause.setText("▶ Play")

    def toggle_play_pause(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.btn_play_pause.setText("▶ Play")
        else:
            if not self.current_song:
                self.play_next()
            else:
                self.media_player.play()
                self.btn_play_pause.setText("⏸ Pause")

    def change_music_volume(self, value):
        self.media_player.audio_set_volume(value)
        self.music_vol_label.setText(f"{value}%")

    def set_audio_channel(self, mode):
        if mode == "left":
            self.media_player.audio_set_channel(3)  # Left channel
        elif mode == "right":
            self.media_player.audio_set_channel(4)  # Right channel
        else:
            self.media_player.audio_set_channel(1)  # Stereo

    def apply_vlc_vocal_eq(self, enable):
        if enable:
            eq = vlc.AudioEqualizer()
            # Scoop vocal frequencies in VLC's standard 10-band equalizer
            eq.set_amp_at_index(-15.0, 3)  # 250 Hz
            eq.set_amp_at_index(-18.0, 4)  # 500 Hz
            eq.set_amp_at_index(-18.0, 5)  # 1 kHz
            eq.set_amp_at_index(-15.0, 6)  # 2 kHz
            eq.set_amp_at_index(-12.0, 7)  # 4 kHz
            self.media_player.set_equalizer(eq)
        else:
            self.media_player.set_equalizer(None)

    def toggle_vocal_eq(self):
        self.vocal_eq_active = not self.vocal_eq_active
        if self.vocal_eq_active:
            self.btn_vocal_eq.setText("🎙️ Vocal Parametric EQ (80-4kHz Cut): ON")
            self.btn_vocal_eq.setObjectName("eqFilterOn")
            self.apply_vlc_vocal_eq(True)
        else:
            self.btn_vocal_eq.setText("🎙️ Vocal Parametric EQ (80-4kHz Cut): OFF")
            self.btn_vocal_eq.setObjectName("eqFilterOff")
            self.apply_vlc_vocal_eq(False)
        self.btn_vocal_eq.setStyle(self.btn_vocal_eq.style())


# ==============================================================================
# 8. APPLICATION ENTRY POINT
# ==============================================================================
def main():
    app = QApplication(sys.argv)

    display_win = VideoDisplayWindow()
    control_win = KTVControlWindow(display_win)

    display_win.show()
    control_win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()