# JetEngine

JetEngine, a lightweight inference engine for the [SDAR](https://jetastra.github.io/SDAR/) series (and other diffusion block decoding models) built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) support both dense and MoE models and Tensor Parallel distributed inference, delivers tons of acceleration compared to the naive implementation.

In our benchmark, we tested the 4B SDAR model with block size 4 (basic acceleration setting) and batch size 128:
- On NVIDIA A800, JetEngine reached 1800+ tokens/second.
- On NVIDIA H200, JetEngine achieved 3700+ tokens/second using FlashAttention-2 + Triton kernels.

This demonstrates that JetEngine can unlock production-level throughput for SDAR models, making it ideal for both research-scale batch inference and real-world deployment scenarios.
## 🚀 New Features
[09/15/2025] Support completely offload the model and kv cache to free memory for RL training
[09/14/2025] Support Hybrid Data Parallel and Tensor Parallel Inference
[09/07/2025] Support [Entropy Bounded sampler](https://arxiv.org/abs/2505.24857)
```python
SamplingParams(temperature=1.0, topk=0, topp=1.0, max_tokens=4096, remasking_strategy="entropy_bounded", block_length=4, denoising_steps=4, eb_threshold=0.6)
```
`eb_threshold` is the $\gamma$ value from the above paper

## Installation
### Environment Setup

```
transformers>=4.52.4
flash-attn
```

For Local Inference:

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Labman42/JetEngine.git
cd JetEngine
pip install .
```
For RL training usage (support DP and TP, managed by accelerate from huggingface):

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Labman42/JetEngine.git
cd JetEngine
git checkout accelerate
pip install .
```

## Manual Download
If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download JetLM/SDAR-1.7B-Chat \
  --local-dir ~/huggingface/SDAR-1.7B-Chat/ \
  --local-dir-use-symlinks False
```

## Quick Start

On default, use all gpus for Data Parallel

```bash
accelerate launch --multi_gpu dist_example.py
```

See `example.py` for usage. The API mirrors vLLM's interface with some differences in the `LLM.generate` method.

## 📬 Contact

For issues or inquiries:
- **Yihan Bian**, University of Maryland, College Park (ybian@umd.edu)
