# 🎙️ Speaker: Advanced Generative TTS & Voice Cloning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker: Supported](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![CUDA: 12.0+](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**Speaker** is a state-of-the-art Generative Text-to-Speech (TTS) platform delivering ultra-realistic, low-latency speech synthesis. Powered by **GLM-TTS**, it supports high-fidelity zero-shot voice cloning with as little as 3 seconds of audio.

<!-- [PLACEHOLDER: Main Demo GIF showing voice cloning process] -->
![Speaker Hero Architecture](./docs/images/architecture_hero.png)

---

## ✨ Core Features

*   **🎭 Zero-Shot Voice Cloning**: Clone any voice perfectly using a brief (3-10s) audio sample.
*   **⚡ Extreme Performance**: Integrated with **vLLM** and **TensorRT** for real-time speedups (up to 6x+ real-time).
*   **🌈 Emotion & Style Control**: RL-enhanced models for expressive speech across various emotions and styles.
*   **🌍 Multi-Language Support**: Optimized for English and Chinese, with multi-dataset support.
*   **📈 Full Stack Monitoring**: Built-in Prometheus metrics and Grafana dashboards for hardware and inference telemetry.
*   **🛠️ Developer-First API**: Fully async FastAPI backend with OpenAPI/Swagger documentation.
*   **🔋 Production Ready**: Pre-configured for Docker, Kubernetes, and NVIDIA Blackwell (RTX 5090) hardware.

---

## 🚀 Tech Stack & Optimizations

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Primary Backend** | **GLM-TTS** | Llama-based LLM + Flow Matching DiT |
| **Inference Engine** | **vLLM** | High-throughput LLM serving via PagedAttention |
| **Vocoder** | **HiFT (TensorRT)** | 3x faster spectrogram-to-audio conversion |
| **API Framework** | **FastAPI** | High-performance asynchronous REST API |
| **Hardware Ops** | **NVIDIA Blackwell** | Optimized for BF16 and FP8 precision |
| **Monitoring** | **Grafana/Prometheus** | Real-time hardware & inference tracking |

---

## 🛠️ Quick Start

### 1. Using Docker Compose (Recommended)

Start the entire stack including the API, Frontend, and Monitoring suite:

```bash
docker-compose up -d
```

| Service | Endpoint |
| :--- | :--- |
| **Frontend UI** | [http://localhost:3012](http://localhost:3012) |
| **API Docs (Swagger)** | [http://localhost:8012/docs](http://localhost:8012/docs) |
| **Grafana Dashboard** | [http://localhost:3333](http://localhost:3333) (admin/admin123) |
| **Prometheus** | [http://localhost:9199](http://localhost:9199) |

### 2. Manual Development Setup

```bash
# Clone the repository
git clone https://github.com/alecKarfonta/speaker.git
cd speaker

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt

# Start the API
./scripts/start_api.sh
```

---

## 🧪 Performance & Validation

We include a comprehensive validation suite to ensure both speed and transcription accuracy.

<!-- [PLACEHOLDER: Terminal recording of run_tts_validation.py execution] -->

```bash
# Run full performance and STT round-trip validation
STT_API_URL=http://<stt-host>:8603/v1/audio/transcriptions ./venv/bin/python3 scripts/run_tts_validation.py
```

### Validation Features:
*   **Numeric Normalization**: Uses `num2words` to correctly validate spoken currency, digits, and dates.
*   **Latency Profiling**: Detailed breakdown of LLM vs. Flow Matching stages.
*   **Real-time Tracking**: Measures RTF (Real-Time Factor) across varying text lengths.

---

## 🖥️ Monitoring (Grafana)

Speaker provides a deep-dive Grafana dashboard for hardware and inference monitoring.

<!-- [PLACEHOLDER: Screenshot of Grafana Dashboard showing RTF and VRAM usage] -->

**Key Metrics Tracked:**
*   **RTF (Real-Time Factor)**: Audio duration / Inference time.
*   **VRAM Utilization**: Broken down by LLM (vLLM) and Flow (DiT) models.
*   **Stage Breakdown**: Millisecond-level latency for LLM, Flow, Phonemizer, and Vocoder.

---

## 🔧 Hardware Configuration

Optimized for **NVIDIA RTX 5090 (Blackwell)**. Configure optimizations in `docker-compose.yml`:

```yaml
environment:
  - GLM_TTS_ENGINE=vllm           # Use vLLM for high performance
  - GLM_TTS_QUANTIZATION=4bit    # Enable 4-bit quantization
  - GLM_TTS_VOCODER_ENGINE=tensorrt # 3x faster vocoder
  - GLM_TTS_COMPILE_FLOW=true     # torch.compile for DiT
```

---

## 🤝 Contributing

Contributions are welcome! Please see [VOICE_LIBRARY_IMPLEMENTATION.md](VOICE_LIBRARY_IMPLEMENTATION.md) for details on adding new backends or optimizing existing ones.

1.  Fork the Repo
2.  Create a Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit Changes (`git commit -m 'Add Features'`)
4.  Push to Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
