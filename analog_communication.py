# --- 1. ABSOLUTE TOP SCRIPT GUARD (Handles folder spaces safely using subprocess) ---
import sys
import os
import subprocess
from streamlit.runtime import exists

if __name__ == "__main__" and not exists():
    env_python = os.path.abspath(os.path.join(".", "env", "Scripts", "python.exe"))
    script_path = os.path.abspath(__file__)
    if os.path.exists(env_python):
        subprocess.run([env_python, "-m", "streamlit", "run", script_path])
    else:
        subprocess.run(["python", "-m", "streamlit", "run", script_path])
    sys.exit()

# --- 2. CORE TOOLBOX IMPORTS ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import wavfile
import io
import time
from sklearn.ensemble import RandomForestClassifier

# --- 3. Page Configuration ---
st.set_page_config(page_title="AM/FM Comm Live Toolbox", layout="wide")
st.title("🤖 AI-Powered AM & FM Telecommunications Toolbox & Spectrum Analyzer")
st.markdown("""
This combined module supports **Amplitude Modulation (AM)** and **Frequency Modulation (FM)** variants, complete with a rolling time-domain oscilloscope, power calculators, and real-time FFT frequency monitoring.
""")

# --- 4. AI Machine Learning Engine (Cached) ---
@st.cache_resource
def train_signal_classifier():
    X, y = [], []
    for _ in range(40):
        n = np.random.normal(0, 1.0, 1000)
        env = np.abs(signal.hilbert(n))
        zcr = np.sum(np.diff(np.sign(n)) != 0) / 1000
        papr = np.max(n**2) / np.mean(n**2) if np.mean(n**2) > 0 else 0
        X.append([np.std(env), zcr, papr])
        y.append(0)
    for _ in range(80):
        t_d = np.linspace(0, 0.05, 1000)
        s = np.cos(2 * np.pi * 300 * t_d)
        env = np.abs(signal.hilbert(s))
        zcr = np.sum(np.diff(np.sign(s)) != 0) / 1000
        papr = np.max(s**2) / np.mean(s**2)
        X.append([np.std(env), zcr, papr])
        y.append(1)
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)
    return model

ai_model = train_signal_classifier()

# --- 5. Helper Functions for DSP Filtering & Power ---
def lowpass_filter(data, cutoff_hz, fs_rate, order=5):
    nyq = 0.5 * fs_rate
    normal_cutoff = cutoff_hz / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def calculate_power(sig):
    """Calculates average power of a signal in Watts and dBW"""
    power_watts = np.mean(sig**2)
    power_dbw = 10 * np.log10(power_watts) if power_watts > 1e-12 else -120
    return power_watts, power_dbw

# --- 6. Sidebar Controls ---
fs = 44100  

st.sidebar.header("🕹️ System Framework Controls")
st.sidebar.subheader("📻 Modulation Type Selection")
mod_scheme = st.sidebar.selectbox(
    "Choose Modulation Scheme", 
    [
        "Standard AM (DSB-TC with Carrier)", 
        "DSB-SC (Suppressed Carrier AM)", 
        "SSB-SC (Single Sideband AM USB)",
        "Frequency Modulation (FM)"
    ]
)

st.sidebar.subheader("✉️ Baseband Message Input")
input_mode = st.sidebar.radio("Input Source", ["Standard Waveforms", "Upload Audio File (.wav)"])

message = np.array([])
t = np.array([])

if input_mode == "Standard Waveforms":
    duration = st.sidebar.slider("Signal Duration (sec)", 0.2, 1.0, 0.5, step=0.1)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal_type = st.sidebar.selectbox("Waveform Shape", ["Sine Wave", "Square Wave", "Triangle Wave"])
    fm = st.sidebar.slider("Message Frequency (Hz)", 200, 1000, 440)
    
    if signal_type == "Sine Wave":
        message = np.sin(2 * np.pi * fm * t)
    elif signal_type == "Square Wave":
        message = signal.square(2 * np.pi * fm * t)
    elif signal_type == "Triangle Wave":
        message = signal.sawtooth(2 * np.pi * fm * t, width=0.5)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Mono .wav", type=["wav"])
    if uploaded_file is not None:
        file_fs, data = wavfile.read(uploaded_file)
        if len(data.shape) > 1: data = data[:, 0]
        if file_fs != fs:
            num_samples = int(len(data) * fs / file_fs)
            message = signal.resample(data, num_samples)
        else:
            message = data.copy()
        if len(message) > fs * 1.5: message = message[:int(fs * 1.5)]
        duration = len(message) / fs
        t = np.linspace(0, duration, len(message), endpoint=False)
        fm = 440
    else:
        st.info("Upload a .wav file. Falling back to 440Hz reference tone.")
        duration = 0.5
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        message = np.sin(2 * np.pi * 440 * t)
        fm = 440

st.sidebar.subheader("🚀 RF Transceiver Settings")
fc = st.sidebar.slider("Carrier Frequency (Hz)", 2000, 12000, 6000)

# Contextual labeling for adjustment parameters
if mod_scheme == "Frequency Modulation (FM)":
    ka = st.sidebar.slider("Frequency Deviation Constant (kf)", 1000, 5000, 3000, step=100)
else:
    ka = st.sidebar.slider("Modulation Index / Scaling (ka)", 0.1, 1.5, 0.8, step=0.05)

snr_db = st.sidebar.slider("Channel Noise (SNR in dB)", -5, 30, 25, step=1)

if np.max(np.abs(message)) > 0:
    message = message / np.max(np.abs(message))

# --- 7. Unified Mathematical Modulation Engine (Transmitter) ---
if mod_scheme == "Standard AM (DSB-TC with Carrier)":
    modulated = (1.0 + ka * message) * np.cos(2 * np.pi * fc * t)
elif mod_scheme == "DSB-SC (Suppressed Carrier AM)":
    modulated = ka * message * np.cos(2 * np.pi * fc * t)
elif mod_scheme == "SSB-SC (Single Sideband AM USB)":
    m_hilbert = np.imag(signal.hilbert(message))
    modulated = ka * (message * np.cos(2 * np.pi * fc * t) - m_hilbert * np.sin(2 * np.pi * fc * t))
elif mod_scheme == "Frequency Modulation (FM)":
    # Compute phase integral for instantaneous frequency integration: phase = 2*pi*kf * integral(m(tau)d_tau)
    phase_integral = np.cumsum(message) / fs
    modulated = np.cos(2 * np.pi * fc * t + 2 * np.pi * ka * phase_integral)

# --- 8. Additive White Gaussian Noise (AWGN) Channel ---
sig_power = np.mean(modulated**2)
snr_linear = 10**(snr_db / 10)
noise_power = sig_power / snr_linear if snr_linear > 0 else 1.0
noise = np.random.normal(0, np.sqrt(noise_power), len(modulated))
received_signal = modulated + noise

# --- 9. Unified Mathematical Demodulation Engine (Receiver) ---
if mod_scheme == "Standard AM (DSB-TC with Carrier)":
    # Envelope Detection using Hilbert Magnitude Transform
    envelope = np.abs(signal.hilbert(received_signal))
    recovered_message = envelope - np.mean(envelope)
elif mod_scheme == "Frequency Modulation (FM)":
    # Demodulate via Hilbert instantaneous phase differentiation
    analytic_signal = signal.hilbert(received_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # Differentiate phase to extract frequency information: f = (1/2pi) * d(phi)/dt
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (1.0 / fs))
    instantaneous_frequency = np.insert(instantaneous_frequency, 0, instantaneous_frequency[0])
    demod_ac = instantaneous_frequency - fc
    cutoff = min(fc - 500, max(fm * 2, 2500))  
    recovered_message = lowpass_filter(demod_ac, cutoff, fs)
else:
    # Coherent / Product Detection for DSB-SC and SSB-SC
    demod_mixed = received_signal * np.cos(2 * np.pi * fc * t)
    cutoff = min(fc - 500, max(fm * 2, 2500))  
    recovered_message = lowpass_filter(demod_mixed, cutoff, fs)

if np.max(np.abs(recovered_message)) > 0:
    recovered_message = recovered_message / np.max(np.abs(recovered_message))

# --- Calculate Signal Powers ---
p_input_w, p_input_db = calculate_power(message)
p_mod_w, p_mod_db = calculate_power(modulated)
p_rx_w, p_rx_db = calculate_power(recovered_message)

# --- 10. AI Diagnostics Calculations ---
live_zcr = np.sum(np.diff(np.sign(received_signal)) != 0) / len(received_signal)
live_papr = np.max(received_signal**2) / np.mean(received_signal**2) if np.mean(received_signal**2) > 0 else 0
live_std_env = np.std(np.abs(signal.hilbert(received_signal)))

features = [[live_std_env, live_zcr, live_papr]]
ai_prediction = ai_model.predict(features)[0]
ai_conf = np.max(ai_model.predict_proba(features)[0]) * 100

# --- 11. UI Dashboard Rendering ---
st.subheader(f"📊 Current Mode: {mod_scheme}")

# System Diagnostics Columns
m1, m2, m3 = st.columns(3)
with m1:
    if snr_db < 2 or ai_prediction == 0:
        st.error(f"🚨 **AI Link Status: Broken Channel**\n\nConfidence: {ai_conf:.1f}%")
    elif snr_db < 12:
        st.warning(f"⚠️ **AI Link Status: Degraded Channel**\n\nConfidence: {ai_conf:.1f}%")
    else:
        st.success(f"👑 **AI Link Status: Clean Signal Matrix**\n\nConfidence: {ai_conf:.1f}%")
with m2:
    st.metric(label="Zero-Crossing Rate (ZCR)", value=f"{live_zcr:.4f}")
with m3:
    st.metric(label="Peak-to-Average Power Ratio (PAPR)", value=f"{live_papr:.2f} dB")

# Power Engine Matrix Status Section
st.markdown("### ⚡ Live System Signal Power Dashboard")
pow_col1, pow_col2, pow_col3 = st.columns(3)
with pow_col1:
    st.metric(label="Baseband Message Power", value=f"{p_input_w:.4f} W", delta=f"{p_input_db:.1f} dBW")
with pow_col2:
    st.metric(label="RF Transmitted Channel Power", value=f"{p_mod_w:.4f} W", delta=f"{p_mod_db:.1f} dBW")
with pow_col3:
    st.metric(label="Recovered Output Power", value=f"{p_rx_w:.4f} W", delta=f"{p_rx_db:.1f} dBW")

st.markdown("---")
col1, col2 = st.columns([1, 1])

# --- COLUMN 1: AUDIO STREAM ARCHITECTURE ---
with col1:
    st.subheader("🔊 Live Continuous Audio Monitor")
    
    def get_wav_bytes(signal_array, sampling_rate):
        byte_io = io.BytesIO()
        wavfile.write(byte_io, sampling_rate, signal_array.astype(np.float32))
        return byte_io.getvalue()

    audio_selection = st.radio(
        "Select Audio Monitoring Channel Stream:",
        ["Mute / Pause Stream", "Listen to Original Baseband Message", "Listen to Demodulated Receiver Output"],
        index=0, horizontal=True
    )
    
    st.markdown("---")
    if audio_selection == "Listen to Original Baseband Message":
        st.caption("🎵 *Streaming Original Input Waveform...*")
        st.audio(get_wav_bytes(message, fs), format="audio/wav", autoplay=True, loop=True)
    elif audio_selection == "Listen to Demodulated Receiver Output":
        st.caption("📻 *Streaming Filtered Receiver Demodulator Output...*")
        st.audio(get_wav_bytes(recovered_message, fs), format="audio/wav", autoplay=True, loop=True)
    else:
        st.info("Stream Muted. Select an active option to start automated loop playback.")

# --- COLUMN 2: TELEMETRY OSCILLOSCOPE & FFT SPECTRUM ---
with col2:
    st.subheader("📈 Time Domain & Frequency Spectrum Viewport")
    live_oscilloscope = st.checkbox("🔄 Activate Moving Telemetry Mode", value=False)
    
    window_size = 600  
    zoom_samples = min(len(t), 1000)
    oscilloscope_placeholder = st.empty()
    
    def compute_fft(sig, sampling_rate):
        N = len(sig)
        fft_vals = np.fft.rfft(sig)
        fft_freqs = np.fft.rfftfreq(N, 1/sampling_rate)
        magnitude = (2.0 / N) * np.abs(fft_vals)
        return fft_freqs, magnitude

    # Plot Layout Generation
    def generate_plots(start_idx, end_idx):
        fig, axs = plt.subplots(4, 1, figsize=(10, 11))
        t_ms = t[start_idx:end_idx] * 1000
        
        # 1. Time Domain - Baseband Message
        axs[0].plot(t_ms, message[start_idx:end_idx], color='#007acc', lw=2)
        axs[0].set_title("1. Input Baseband Message Signal (Time Domain: m(t))")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True, linestyle=":")
        
        # 2. Time Domain - Modulated RF Waveform
        axs[1].plot(t_ms, received_signal[start_idx:end_idx], color='#e056fd', lw=1)
        axs[1].set_title(f"2. Modulated Signal Channel (Time Domain: {mod_scheme})")
        axs[1].set_ylabel("Amplitude")
        axs[1].grid(True, linestyle=":")
        
        # 3. Time Domain - Reconstructed Output
        axs[2].plot(t_ms, recovered_message[start_idx:end_idx], color='#20bf6b', lw=2, linestyle='--')
        axs[2].set_title("3. Output Demodulated Filtered Waveform (Time Domain: y(t))")
        axs[2].set_ylabel("Amplitude")
        axs[2].grid(True, linestyle=":")
        
        # 4. Frequency Domain Spectrum Analysis (FFT)
        freq_in, mag_in = compute_fft(message[start_idx:end_idx], fs)
        freq_mod, mag_mod = compute_fft(received_signal[start_idx:end_idx], fs)
        
        # Set dynamic view limit around the carrier frequency to catch sidebands cleanly (+/- 5kHz)
        max_view_freq = min(fc + 5000, fs/2)
        
        axs[3].plot(freq_in, mag_in, color='#007acc', lw=1.5, label='Input Message Spectrum')
        axs[3].plot(freq_mod, mag_mod, color='#e056fd', lw=1.2, alpha=0.8, label='Modulated RF Spectrum')
        axs[3].set_title("4. Frequency Spectrum Density Analysis (FFT Viewport)")
        axs[3].set_xlabel("Frequency (Hz)")
        axs[3].set_ylabel("Magnitude")
        axs[3].set_xlim(0, max_view_freq)
        axs[3].legend(loc="upper right")
        axs[3].grid(True, linestyle=":")
        
        plt.tight_layout()
        return fig

    # Execution Animation Controller
    if live_oscilloscope:
        st.caption("📺 *Streaming telemetry frames... Adjust parameters in the sidebar to view transformations.*")
        step_increment = 35
        while live_oscilloscope:
            for frame_start in range(0, len(t) - window_size, step_increment):
                if not live_oscilloscope:
                    break
                end_idx = frame_start + window_size
                fig = generate_plots(frame_start, end_idx)
                oscilloscope_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.005)
    else:
        fig = generate_plots(0, zoom_samples)
        oscilloscope_placeholder.pyplot(fig)
        plt.close(fig)