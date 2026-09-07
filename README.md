🎙️ KTV Vocal Engine & Low-Latency Microphone PassthroughA high-performance Python-based KTV (Karaoke) and real-time vocal processing application. Built with PyQt6, python-vlc, and sounddevice, this application features a dual-window architecture, real-time microphone audio processing with delay/decay DSP, parametric vocal EQ filtering, and dynamic queue management.  ✨ Key Features🚀 Low-Latency Audio Passthrough: Operates with dynamic driver hints (latency='low') and small buffer blocks (64 to 128 samples) to minimize delay.  🎛️ Dual-Microphone Input & DSP Echo/Reverb: Independent gain controls for up to two microphones with adjustable real-time feedback echo delay and decay.  🎙️ Parametric Vocal EQ Filter: Custom frequency attenuation (cutting 80 Hz–4 kHz vocal ranges) via SciPy DSP and a built-in 10-band VLC equalizer preset to suppress vocals in track recordings.  🖥️ Dual-Window KTV Display:Control Panel Window: Library search browser, drag-and-drop queue reordering, track mode selection (Stereo, Left-Channel Instrumental, Right-Channel Vocal), and microphone settings.  Video Display Window: Standalone playback window intended for secondary screens or projector output, featuring double-click/hotkey fullscreen toggle.  🔄 Smart Autoplay & Queue Management: Supports drag-and-drop reordering, sequential queue autoplay, and an automatic fallback mode to play random tracks when the queue ends.  🛠️ Prerequisites & InstallationRequirementsPython: 3.9 or higher  VLC Media Player (64-bit): Must be installed on your system (default fallback path configured for C:\Program Files\VideoLAN\VLC on Windows).  Installation StepsClone the repository or save the source code:Bashgit clone https://github.com/your-username/ktv-vocal-engine.git
cd ktv-vocal-engine
Install Python dependencies:Bashpip install PyQt6 python-vlc sounddevice numpy scipy
```[cite: 1]

🚀 Running the ApplicationLaunch the main Python application script:Bashpython main.py
```[cite: 1]

Upon launching, the app opens two windows[cite: 1]:
1. **KTV Video Display:** The dedicated window for rendering video content[cite: 1].
2. **KTV Control Panel:** The main hub to control playback, adjust sliders, manage audio devices, and queue up tracks[cite: 1].

---

## ⌨️ Controls & Hotkeys

| Action | Shortcut / Trigger |
| :--- | :--- |
| **Toggle Fullscreen Video** | Press `F` key or double-click inside the Video Display window[cite: 1] |
| **Skip Song** | Press `N`, `Ctrl + Right Arrow`, or the `Media Next` key on your keyboard[cite: 1] |
| **Reorder Queue** | Drag and drop items inside the **Selected Songs Queue** box[cite: 1] |

---

## ⚙️ Audio Track Modes & KTV Features

- **Audio Track Switching:** Built-in buttons quickly switch between full **Stereo**, **Music (Left)** channel (instrumentals), and **Vocal (Right)** channel for traditional dual-channel KTV media files[cite: 1].
- **Vocal Parametric EQ Cut:** Toggle the **🎙️ Vocal Parametric EQ** button to engage VLC's internal 10-band equalizer, scooping core speech frequencies (170 Hz–3 kHz) when using standard stereo tracks without separate channels[cite: 1].
- **Auto-Scan Directory:** On launch, the application scans your local `Downloads` directory for supported media files (`.mp4`, `.mkv`, `.avi`, `.mov`)[cite: 1]. Use the **📁 Change Folder** button to point the browser to your personal song library[cite: 1].

---

## 📄 License

Distributed under the MIT License.
