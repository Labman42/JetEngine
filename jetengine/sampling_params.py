from dataclasses import dataclass
from typing import Literal

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # Block Diffusion Parameters
    block_length: int = 4
    denoising_steps: int = 4
    dynamic_threshold: float = 0.8  # opt27: 0.9→0.8, +0.9% pass@1 on MATH-500 gen4
    eb_threshold: float = 0.35
    topk: int = 0
    topp: float = 1
    repetition_penalty: float = 1.0
    # Causal waterfall: per-position log-prob floor. cw_tau = -0.35 ⇒ only
    # commit positions with P(argmax) > exp(-0.35) ≈ 0.70. Committed prefix
    # must be contiguous leftmost. At T=1.0 this only fires on syntactically
    # forced tokens (digits, delimiters, common function words) — the positions
    # where the joint factorization is accurate because marginals are peaked.
    cw_tau: float = -0.35
    # Logit salience z-score threshold (outlier detection in vocab space).
    # 10 is very strict → rarely fires on diffuse distributions at T=1.0.
    lsz_threshold: float = 10.0
    # Relative-top: alpha × seq-max-P(argmax) with absolute floor 0.5 inside.
    rel_top_alpha: float = 0.95
    remasking_strategy: Literal[
        'sequential', 'low_confidence_static', 'low_confidence_dynamic',
        'entropy_bounded', 'random',
        # New strategies for remasking research
        'highest_confidence', 'anti_sequential', 'top2_margin', 'least_entropy',
        # Joint-distribution aware multi-commit strategies (opt27)
        'causal_waterfall', 'logit_salience', 'relative_top', 'dynamic_contiguous',
        'confidence_chain', 'dynamic_pmax',
    ] = 'low_confidence_static'
    stop_words: list[int] | None = None
