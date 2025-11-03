# JetEngine
## ⚠️ Development Notice

This project is under **active development (Accelerate branch)**.  
If you encounter any issues on the `main` branch, please [open an issue](../../issues) to remind me to fix them.  
Your feedback is greatly appreciated!

## 📢 Kind Notice

JetEngine is a personal project I started during my work on [SDAR](https://jetastra.github.io/SDAR/), inspired by the excellent open-source project [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

I built JetEngine for fun and to demonstrate the potential speedups of SDAR vs. standard AR models.

Recently, I’ve received feedback and interest from others about adapting this inference engine for RL training. I’m happy to hear that my work may be useful! Since I’m new to the open-source world, I may sometimes make mistakes—please bear with me 🙏. I truly welcome advice and suggestions from the community.

Currently, most active development is happening in the Accelerate branch, which focuses on RL training for SDAR. As this is a personal project, I cannot thoroughly test every scenario or provide complete documentation, but I will update this README as new features are added.

✨ If you are also working on SDAR RL training, or if you are experienced in inference engines / ML systems, your feedback and contributions would mean a lot to me. That’s the beauty of open source.

As a side note, I’m also considering experimenting with fast-dllm to support models like Dream/Llada in the future, though this is not my first priority. If this interests you, please feel free to reach out!

JetEngine, a lightweight inference engine for the [SDAR](https://jetastra.github.io/SDAR/) series (and other diffusion block decoding models) built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) support both dense and MoE models and Tensor Parallel distributed inference, delivers tons of acceleration compared to the naive implementation.

## ⚡ Naive speedtest results

In our benchmark, we tested the 4B SDAR model with block size 4 (basic acceleration setting) and batch size 128:
- On NVIDIA A800, JetEngine reached 1800+ tokens/second.
- On NVIDIA H200, JetEngine achieved 3700+ tokens/second using FlashAttention-2 + Triton kernels.

This demonstrates that JetEngine can unlock production-level throughput for SDAR models, making it ideal for both research-scale batch inference and real-world deployment scenarios.
## 🚀 New Features
[11/03/2025] Support Llada in a very interesting way, please check the `accelerate` branch

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

```bash
python example.py
```

See `example.py` for usage. The API mirrors vLLM's interface with some differences in the `LLM.generate` method.

## 📬 Contact

For issues or inquiries:
- **Yihan Bian**, University of Maryland, College Park (ybian@umd.edu)
