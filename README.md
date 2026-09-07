# 🎙️ KTV Vocal Engine & Low-Latency Microphone Passthrough

A high-performance Python-based KTV (Karaoke) and real-time vocal processing application[cite: 1]. Built with **PyQt6**, **python-vlc**, and **sounddevice**, this application features a dual-window architecture, real-time microphone audio processing with delay/decay DSP, parametric vocal EQ filtering, and dynamic queue management[cite: 1].

---

## ✨ Key Features

- **🚀 Low-Latency Audio Passthrough:** Operates with dynamic driver hints (`latency='low'`) and small buffer blocks (64 to 128 samples) to minimize delay[cite: 1].
- **🎛️ Dual-Microphone Input & DSP Echo/Reverb:** Independent gain controls for up to two microphones with adjustable real-time feedback echo delay and decay[cite: 1].
- **🎙️ Parametric Vocal EQ Filter:** Custom frequency attenuation (cutting 80 Hz–4 kHz vocal ranges) via SciPy DSP and a built-in 10-band VLC equalizer preset to suppress vocals in track recordings[cite: 1].
- **🖥️ Dual-Window KTV Display:**
  - **Control Panel Window:** Library search browser, drag-and-drop queue reordering, track mode selection (Stereo, Left-Channel Instrumental, Right-Channel Vocal), and microphone settings[cite: 1].
  - **Video Display Window:** Standalone playback window intended for secondary screens or projector output, featuring double-click/hotkey fullscreen toggle[cite: 1].
- **🔄 Smart Autoplay & Queue Management:** Supports drag-and-drop reordering, sequential queue autoplay, and an automatic fallback mode to play random tracks when the queue ends[cite: 1].

---

## 🛠️ Prerequisites & Installation

### Requirements
- **Python:** 3.9 or higher[cite: 1]
- **VLC Media Player (64-bit):** Must be installed on your system (default fallback path configured for `C:\Program Files\VideoLAN\VLC` on Windows)[cite: 1].

### Installation Steps

1. **Clone the repository or save the source code:**
   ```bash
   git clone [https://github.com/Brian-Hou-818/karaoke.git](https://github.com/your-username/ktv-vocal-engine.git)
   cd ktv-vocal-engine


   pip install PyQt6 python-vlc sounddevice numpy scipy
