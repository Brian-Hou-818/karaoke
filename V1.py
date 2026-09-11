import os
import sys
import ctypes
import threading
import sounddevice as sd
import numpy as np

# ==============================================================================
# 1. WINDOWS DLL PATCH & SETUP FOR PYTHON-VLC
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
    QListWidgetItem, QSlider, QComboBox, QAbstractItemView
)
from PyQt6.QtCore import Qt


# ==============================================================================
# 3. DRAG AND DROP QUEUE WIDGET
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
# 4. FULL-DUPLEX DUAL-MIC AUDIO ENGINE (REVERTED TO WORKING VERSION)
# ==============================================================================
class DualMicEngine:
    """ Hardware-synchronized full-duplex streams """

    def __init__(self):
        self.block_size = 128  # Clean balance between latency and stability

        self.mic1_enabled = False
        self.mic1_dev_id = None
        self.mic1_gain = 1.0

        self.mic2_enabled = False
        self.mic2_dev_id = None
        self.mic2_gain = 1.0

        self.stream1 = None
        self.stream2 = None

    def _duplex_callback_1(self, indata, outdata, frames, time, status):
        """ Hardware-synchronized callback for Mic 1 """
        if self.mic1_enabled and indata is not None:
            audio = indata[:, 0] if indata.ndim > 1 else indata
            mixed = audio * self.mic1_gain

            np.clip(mixed, -1.0, 1.0, out=mixed)

            if outdata.ndim > 1:
                for col in range(outdata.shape[1]):
                    outdata[:, col] = mixed
            else:
                outdata[:] = mixed
        else:
            outdata.fill(0)

    def _duplex_callback_2(self, indata, outdata, frames, time, status):
        """ Hardware-synchronized callback for Mic 2 """
        if self.mic2_enabled and indata is not None:
            audio = indata[:, 0] if indata.ndim > 1 else indata
            mixed = audio * self.mic2_gain

            np.clip(mixed, -1.0, 1.0, out=mixed)

            if outdata.ndim > 1:
                for col in range(outdata.shape[1]):
                    outdata[:, col] = mixed
            else:
                outdata[:] = mixed
        else:
            outdata.fill(0)

    def toggle_mic1(self, dev_id, gain):
        self.mic1_gain = gain
        if self.mic1_enabled:
            self.mic1_enabled = False
            if self.stream1:
                self.stream1.stop()
                self.stream1.close()
                self.stream1 = None
            return False
        else:
            self.mic1_dev_id = dev_id
            try:
                in_info = sd.query_devices(dev_id)
                out_info = sd.query_devices(kind='output')

                sr = int(in_info['default_samplerate'])
                in_ch = min(1, int(in_info['max_input_channels']))
                out_ch = min(2, int(out_info['max_output_channels']))

                self.stream1 = sd.Stream(
                    device=(dev_id, None),  # Direct mic input, Default output
                    samplerate=sr,
                    blocksize=self.block_size,
                    channels=(in_ch, out_ch),
                    dtype='float32',
                    callback=self._duplex_callback_1
                )
                self.stream1.start()
                self.mic1_enabled = True
                return True
            except Exception as e:
                print(f"Mic 1 Start Error: {e}")
                self.mic1_enabled = False
                return False

    def toggle_mic2(self, dev_id, gain):
        self.mic2_gain = gain
        if self.mic2_enabled:
            self.mic2_enabled = False
            if self.stream2:
                self.stream2.stop()
                self.stream2.close()
                self.stream2 = None
            return False
        else:
            self.mic2_dev_id = dev_id
            try:
                in_info = sd.query_devices(dev_id)
                out_info = sd.query_devices(kind='output')

                sr = int(in_info['default_samplerate'])
                in_ch = min(1, int(in_info['max_input_channels']))
                out_ch = min(2, int(out_info['max_output_channels']))

                self.stream2 = sd.Stream(
                    device=(dev_id, None),
                    samplerate=sr,
                    blocksize=self.block_size,
                    channels=(in_ch, out_ch),
                    dtype='float32',
                    callback=self._duplex_callback_2
                )
                self.stream2.start()
                self.mic2_enabled = True
                return True
            except Exception as e:
                print(f"Mic 2 Start Error: {e}")
                self.mic2_enabled = False
                return False

    def set_mic1_gain(self, gain):
        self.mic1_gain = gain

    def set_mic2_gain(self, gain):
        self.mic2_gain = gain

    def stop_all(self):
        if self.stream1:
            self.stream1.stop()
            self.stream1.close()
        if self.stream2:
            self.stream2.stop()
            self.stream2.close()


def disable_windows_audio_ducking():
    """ Disables automatic Windows audio attenuation during mic activation """
    if sys.platform.startswith('win'):
        try:
            ctypes.windll.avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(ctypes.c_ulong(0)))
        except Exception as e:
            print(f"Audio thread boost note: {e}")


# ==============================================================================
# 5. MAIN GUI APPLICATION
# ==============================================================================
class KTVDesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        disable_windows_audio_ducking()

        self.setWindowTitle("KTV Multi-Mic Player & Drag-and-Drop Song Picker")
        self.setGeometry(80, 80, 1400, 820)

        # App State
        self.song_library = []
        self.selected_queue = []
        self.current_song = None

        # Audio Engine
        self.audio_engine = DualMicEngine()

        # VLC Engine with DirectSound flags to prevent conflicts with WASAPI
        self.vlc_instance = vlc.Instance('--aout=directsound')
        self.media_player = self.vlc_instance.media_player_new()

        self.init_ui()
        self.populate_mic_devices()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; }
            QLabel { color: #f8fafc; font-size: 12px; }
            QComboBox { 
                background-color: #1e293b; color: white; 
                border: 1px solid #334155; border-radius: 4px; padding: 3px;
                font-size: 11px;
            }
            QLineEdit { 
                background-color: #1e293b; color: white; 
                border: 1px solid #334155; border-radius: 5px; 
                padding: 6px; font-size: 12px;
            }
            QListWidget { 
                background-color: #1e293b; color: #f8fafc; 
                border: 1px solid #334155; border-radius: 6px; padding: 3px;
                font-size: 12px;
            }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #334155; }
            QListWidget::item:hover { background-color: #334155; }
            QListWidget::item:selected { background-color: #3b82f6; border-radius: 4px; }
            QPushButton { 
                background-color: #334155; color: white; border: none; 
                padding: 6px 10px; border-radius: 5px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #475569; }
            QPushButton#primaryBtn { background-color: #2563eb; }
            QPushButton#primaryBtn:hover { background-color: #1d4ed8; }
            QPushButton#playBtn { background-color: #16a34a; }
            QPushButton#playBtn:hover { background-color: #15803d; }
            QPushButton#micBtn { background-color: #dc2626; }
            QPushButton#micBtn:hover { background-color: #b91c1c; }
            QSlider::groove:horizontal {
                border: 1px solid #334155; height: 5px; 
                background: #1e293b; border-radius: 2px;
            }
            QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #f8fafc; width: 12px; 
                margin-top: -4px; margin-bottom: -4px; border-radius: 6px;
            }
        """)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # LEFT SIDE: VIDEO PLAYER & CONTROLS
        left_box = QVBoxLayout()

        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: #000000; border-radius: 8px;")
        left_box.addWidget(self.video_frame, stretch=1)

        # Playback Controls
        controls_row1 = QHBoxLayout()

        self.btn_open = QPushButton("📁 Load Folder")
        self.btn_open.setObjectName("primaryBtn")
        self.btn_open.clicked.connect(self.select_folder)
        controls_row1.addWidget(self.btn_open)

        self.btn_play_pause = QPushButton("⏸ Pause")
        self.btn_play_pause.setObjectName("playBtn")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        controls_row1.addWidget(self.btn_play_pause)

        self.btn_skip = QPushButton("⏭ Skip")
        self.btn_skip.clicked.connect(self.play_next)
        controls_row1.addWidget(self.btn_skip)

        controls_row1.addSpacing(10)
        controls_row1.addWidget(QLabel("Audio:"))

        self.btn_stereo = QPushButton("Stereo")
        self.btn_stereo.clicked.connect(lambda: self.set_audio_channel("stereo"))
        controls_row1.addWidget(self.btn_stereo)

        self.btn_left = QPushButton("Music (Left)")
        self.btn_left.clicked.connect(lambda: self.set_audio_channel("left"))
        controls_row1.addWidget(self.btn_left)

        self.btn_right = QPushButton("Vocal (Right)")
        self.btn_right.clicked.connect(lambda: self.set_audio_channel("right"))
        controls_row1.addWidget(self.btn_right)

        left_box.addLayout(controls_row1)

        # Music Volume Slider
        controls_row2 = QHBoxLayout()
        controls_row2.addWidget(QLabel("🔊 Music Vol:"))
        self.music_vol_label = QLabel("80%")
        self.music_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.music_vol_slider.setRange(0, 100)
        self.music_vol_slider.setValue(80)
        self.music_vol_slider.valueChanged.connect(self.change_music_volume)
        controls_row2.addWidget(self.music_vol_slider)
        controls_row2.addWidget(self.music_vol_label)
        left_box.addLayout(controls_row2)

        # Mic 1 Controls
        mic1_row = QHBoxLayout()
        mic1_row.addWidget(QLabel("🎙️ Mic 1:"))
        self.mic1_combo = QComboBox()
        mic1_row.addWidget(self.mic1_combo, stretch=1)

        self.btn_mic1_toggle = QPushButton("Mic 1 OFF")
        self.btn_mic1_toggle.setObjectName("micBtn")
        self.btn_mic1_toggle.clicked.connect(self.toggle_mic1)
        mic1_row.addWidget(self.btn_mic1_toggle)

        mic1_row.addWidget(QLabel("Vol:"))
        self.mic1_vol_label = QLabel("100%")
        self.mic1_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic1_vol_slider.setRange(0, 200)
        self.mic1_vol_slider.setValue(100)
        self.mic1_vol_slider.valueChanged.connect(self.change_mic1_volume)
        mic1_row.addWidget(self.mic1_vol_slider)
        mic1_row.addWidget(self.mic1_vol_label)
        left_box.addLayout(mic1_row)

        # Mic 2 Controls
        mic2_row = QHBoxLayout()
        mic2_row.addWidget(QLabel("🎙️ Mic 2:"))
        self.mic2_combo = QComboBox()
        mic2_row.addWidget(self.mic2_combo, stretch=1)

        self.btn_mic2_toggle = QPushButton("Mic 2 OFF")
        self.btn_mic2_toggle.setObjectName("micBtn")
        self.btn_mic2_toggle.clicked.connect(self.toggle_mic2)
        mic2_row.addWidget(self.btn_mic2_toggle)

        mic2_row.addWidget(QLabel("Vol:"))
        self.mic2_vol_label = QLabel("100%")
        self.mic2_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic2_vol_slider.setRange(0, 200)
        self.mic2_vol_slider.setValue(100)
        self.mic2_vol_slider.valueChanged.connect(self.change_mic2_volume)
        mic2_row.addWidget(self.mic2_vol_slider)
        mic2_row.addWidget(self.mic2_vol_label)
        left_box.addLayout(mic2_row)

        main_layout.addLayout(left_box, stretch=3)

        # RIGHT SIDE: QUEUE & BROWSER
        right_box = QVBoxLayout()

        self.now_playing_label = QLabel("<b>Now Playing:</b> None")
        self.now_playing_label.setStyleSheet("color: #60a5fa; margin-bottom: 2px;")
        right_box.addWidget(self.now_playing_label)

        right_box.addWidget(QLabel("<b>Selected Songs Queue</b> <i>(Drag to reorder)</i>"))

        self.queue_widget = DraggableQueueList()
        self.queue_widget.reorder_callback = self.on_queue_reordered
        right_box.addWidget(self.queue_widget, stretch=1)

        btn_remove = QPushButton("❌ Remove Selected")
        btn_remove.clicked.connect(self.remove_from_queue)
        right_box.addWidget(btn_remove)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setStyleSheet("background-color: #334155; margin: 6px 0px;")
        right_box.addWidget(divider)

        right_box.addWidget(QLabel("<b>🔍 Browse Song Library</b>"))

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search title or artist...")
        self.search_bar.textChanged.connect(self.filter_songs)
        right_box.addWidget(self.search_bar)

        self.library_list_widget = QListWidget()
        self.library_list_widget.itemDoubleClicked.connect(self.add_selected_song_from_browser)
        right_box.addWidget(self.library_list_widget, stretch=2)

        self.btn_add_browser = QPushButton("➕ Add to Queue")
        self.btn_add_browser.setObjectName("primaryBtn")
        self.btn_add_browser.clicked.connect(self.add_selected_song_from_browser)
        right_box.addWidget(self.btn_add_browser)

        main_layout.addLayout(right_box, stretch=1)

    def populate_mic_devices(self):
        self.mic1_combo.clear()
        self.mic2_combo.clear()
        devices = sd.query_devices()

        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = f"[{idx}] {dev['name']}"
                self.mic1_combo.addItem(name, userData=idx)
                self.mic2_combo.addItem(name, userData=idx)

        if self.mic2_combo.count() > 1:
            self.mic2_combo.setCurrentIndex(1)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select KTV Songs Folder")
        if folder_path:
            self.song_library.clear()
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                        full_path = os.path.join(root, file)
                        song_name = os.path.splitext(file)[0]
                        self.song_library.append({'name': song_name, 'path': full_path})

            self.populate_library_list(self.song_library)

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

        if not self.media_player.is_playing() and len(self.selected_queue) == 1:
            self.play_next()

    def remove_from_queue(self):
        selected_row = self.queue_widget.currentRow()
        if selected_row >= 0 and selected_row < len(self.selected_queue):
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

            self.now_playing_label.setText(f"<b>Now Playing:</b> {self.current_song['name']}")

            media = self.vlc_instance.media_new(self.current_song['path'])
            self.media_player.set_media(media)

            if sys.platform.startswith('win'):
                self.media_player.set_hwnd(int(self.video_frame.winId()))

            self.media_player.play()
            self.media_player.audio_set_volume(self.music_vol_slider.value())
            self.btn_play_pause.setText("⏸ Pause")
        else:
            self.media_player.stop()
            self.now_playing_label.setText("<b>Now Playing:</b> Queue Finished")
            self.btn_play_pause.setText("▶ Play")

    def toggle_play_pause(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.btn_play_pause.setText("▶ Play")
        else:
            if not self.current_song and self.selected_queue:
                self.play_next()
            else:
                self.media_player.play()
                self.btn_play_pause.setText("⏸ Pause")

    def change_music_volume(self, value):
        self.media_player.audio_set_volume(value)
        self.music_vol_label.setText(f"{value}%")

    def toggle_mic1(self):
        dev_id = self.mic1_combo.currentData()
        gain = self.mic1_vol_slider.value() / 100.0
        if dev_id is not None:
            is_on = self.audio_engine.toggle_mic1(dev_id, gain)
            if is_on:
                self.btn_mic1_toggle.setText("Mic 1 ON")
                self.btn_mic1_toggle.setStyleSheet("background-color: #16a34a;")
            else:
                self.btn_mic1_toggle.setText("Mic 1 OFF")
                self.btn_mic1_toggle.setStyleSheet("background-color: #dc2626;")

    def change_mic1_volume(self, value):
        self.mic1_vol_label.setText(f"{value}%")
        self.audio_engine.set_mic1_gain(value / 100.0)

    def toggle_mic2(self):
        dev_id = self.mic2_combo.currentData()
        gain = self.mic2_vol_slider.value() / 100.0
        if dev_id is not None:
            is_on = self.audio_engine.toggle_mic2(dev_id, gain)
            if is_on:
                self.btn_mic2_toggle.setText("Mic 2 ON")
                self.btn_mic2_toggle.setStyleSheet("background-color: #16a34a;")
            else:
                self.btn_mic2_toggle.setText("Mic 2 OFF")
                self.btn_mic2_toggle.setStyleSheet("background-color: #dc2626;")

    def change_mic2_volume(self, value):
        self.mic2_vol_label.setText(f"{value}%")
        self.audio_engine.set_mic2_gain(value / 100.0)

    def set_audio_channel(self, mode):
        if mode == "stereo":
            self.media_player.audio_set_channel(1)
        elif mode == "left":
            self.media_player.audio_set_channel(3)
        elif mode == "right":
            self.media_player.audio_set_channel(4)

    def closeEvent(self, event):
        self.audio_engine.stop_all()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KTVDesktopApp()
    window.show()
    sys.exit(app.exec())