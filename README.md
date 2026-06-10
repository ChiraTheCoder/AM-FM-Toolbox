# 🤖 AI-Powered AM & FM Telecommunications Toolbox & Spectrum Analyzer

An interactive, web-based digital signal processing (DSP) and telecommunications simulation platform built entirely in Python using **Streamlit**. This application visually models **Amplitude Modulation (AM)** and **Frequency Modulation (FM)** architectures in real-time, coupled with live signal power tracking, a rolling time-domain oscilloscope, an FFT spectrum analyzer, and a machine learning diagnostics layer.

Live web access via Streamlit Community Cloud allows users to experiment with radio frequency (RF) telemetry and signal transmission without specialized laboratory hardware.

---

## 🚀 Key Features

*   **Dual Modulation Topologies:** Supports Amplitude Modulation (including Standard AM with Carrier, DSB-SC, and SSB-SC Upper Sideband) and Frequency Modulation (FM).
*   **Time-Domain Oscilloscope:** A rolling telemetry window that visualizes the baseband message signal, the modulated RF signal transmitted over the air (with noise), and the reconstructed/demodulated receiver output.
*   **FFT Frequency Spectrum Analyzer:** Computes and plots a real-time Fast Fourier Transform (FFT) to monitor carrier spikes, sideband splits, and spectral density distribution.
*   **Live Signal Power Dashboard:** Instantly tracks and prints average root-mean-square (RMS) power for input, transmitted, and recovered waves in both **Watts** and **dBW**.
*   **AI Link Status Diagnostics:** Uses a cached **Random Forest Classifier** to analyze zero-crossing rates (ZCR) and peak-to-average power ratios (PAPR) to evaluate if the transmission link is *Clean*, *Degraded*, or *Broken*.
*   **Dynamic Inputs:** Supports standard generated waveforms (Sine, Square, Triangle) with adjustable frequencies, or custom mono `.wav` audio file uploads.

---

## 🛠️ System Architecture & Math Model

The backend architecture maps closely out to physical transceiver hardware mechanisms:
1.  **Transmitter:** Numerical time-integration ($\int m(\tau) d\tau$) handles FM angle deviation, while Hilbert transforms simulate SSB analytic signals.
2.  **Channel:** Simulates an Additive White Gaussian Noise (AWGN) medium with adjustable SNR (dB) scales.
3.  **Receiver:** Implements diode-style Envelope Detection for standard AM, synchronous Coherent Product Detection for suppressed carriers, and instantaneous phase differentiation ($d\phi/dt$) for FM signal recovery.

---

## 📦 Local Setup Instructions

To run this telecommunications simulator on your local machine using VS Code, follow these quick steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
