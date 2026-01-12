# Speaker: High-Performance Generative Speech Synthesis

Speaker is a specialized platform for ultra-realistic Text-to-Speech synthesis and zero-shot voice cloning. It is designed around the philosophy that speech should feel natural, identity should be portable, and inference should be fast enough for real-time applications.

At its core, Speaker uses the GLM-TTS architecture—a two-stage generative model combining a Llama-based language model for semantic understanding with a flow-matching diffusion transformer for acoustic synthesis. This combination allows for capturing the nuance of a human voice with as little as 3 seconds of reference audio.

---

## Technical Foundation

Most TTS solutions are built as black boxes. Speaker is designed with visibility and high-performance hardware in mind.

### The Inference Pipeline
The synthesis process is split into distinct, optimized stages:
1.  **Frontend**: Text is normalized and converted into semantic tokens.
2.  **LLM (Llama)**: Semantic tokens are transformed into speech tokens. We utilize **vLLM** for this stage, leveraging PagedAttention to achieve high throughput and low latency.
3.  **Flow Matching (DiT)**: Acoustic tokens are generated via a diffusion process. On Blackwell hardware (RTX 5090), this process is optimized with specialized dtypes and torch.compile.
4.  **Vocoder (HiFT)**: Mel-spectrograms are converted into high-fidelity audio. We use a **TensorRT** implementation of the HiFT vocoder, which provides a 3x speedup over standard PyTorch execution.

### Hardware Optimization (NVIDIA Blackwell)
Speaker is preconfigured to take full advantage of modern GPU architectures. This includes:
*   **Precision**: Support for FP8 and BF16 dtypes to maximize throughput on Ada and Blackwell architectures.
*   **JIT Optimization**: Aggressive use of torch.compile for non-autoregressive components.
*   **Quantization**: Built-in 4-bit and 8-bit quantization modes for the LLM stage to minimize VRAM footprint without sacrificing synthesis quality.

---

## Performance Telemetry

Understanding the "black box" of generative speech is critical. Speaker includes a first-class monitoring stack based on Prometheus and Grafana.

Instead of generic health checks, we track the engine's cognitive load:
*   **Real-Time Factor (RTF)**: The ratio of generated audio duration to inference time.
*   **Latency Breakdown**: Per-stage millisecond tracking for the LLM, Flow, and Vocoder.
*   **VRAM Allocation**: Real-time tracking of memory usage, essential for balancing multiple models on a single GPU.

---

## Validation and Reliability

To ensure consistent output quality, we maintain an automated validation suite (`scripts/run_tts_validation.py`) that performs round-trip STT (Speech-to-Text) verification.

A unique challenge in TTS validation is "orthographic drift"—where numbers or symbols are spoken correctly but transcribed differently (e.g., "$47.50" vs "forty-seven dollars and fifty cents"). Our validation engine utilizes `num2words` to normalize these discrepancies, ensuring that similarity scores reflect actual synthesis accuracy rather than formatting variations.

---

## Deployment

### Docker Environment
The simplest way to stand up the full stack is via Docker Compose:

```bash
docker compose up -d --build
```

Service mapping:
*   **Web Interface**: Port 3012
*   **API (OpenAPI)**: Port 8012
*   **Monitoring (Grafana)**: Port 3333

### Component Configuration
Advanced engine settings are managed through environment variables in the compose file. Key toggles include switching between Transformers and vLLM engines, enabling TensorRT vocoders, and adjusting flow-matching steps.

---

## Project Structure

*   `app/`: Core FastAPI application and backend implementations.
*   `GLM-TTS/`: Model definitions and weights.
*   `frontend/`: React-based management UI.
*   `k8s/`: Kubernetes manifest for at-scale deployment.
*   `scripts/`: Utility scripts for validation, metrics, and benchmarking.

---

## License

This project is licensed under the MIT License. Contributions focused on model optimization or hardware acceleration are always welcome.
