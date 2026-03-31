"""
Unit test for JetEngine postprocess correctness.
Tests that postprocess_unify produces identical results to the reference
postprocess_loop implementation for a synthetic batch.

Run on GPU node:
  CUDA_VISIBLE_DEVICES=0 python tests/test_correctness.py
"""
import sys
import os
import copy
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jetengine.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine.engine.scheduler import Scheduler
from jetengine.config import Config


def make_test_config():
    """Create a minimal Config for testing (no model loading)."""
    # We need a Config object. Since Config.__init__ loads hf_config,
    # we mock it minimally.
    class FakeHFConfig:
        model_type = "sdar"
        hidden_size = 256
        num_hidden_layers = 2
        num_attention_heads = 4
        num_key_value_heads = 4
        head_dim = 64
        intermediate_size = 512
        vocab_size = 152064
        max_position_embeddings = 4096
        rms_norm_eps = 1e-6
        rope_theta = 10000

    class FakeConfig:
        model = "fake"
        hf_config = FakeHFConfig()
        max_num_seqs = 64
        max_num_batched_tokens = 4096
        max_model_len = 4096
        eos = 151643
        mask_token_id = 151669
        block_length = 4
        kvcache_block_size = 256
        gpu_memory_utilization = 0.9
        enforce_eager = True
        num_kvcache_blocks = 100
        torch_dtype = torch.bfloat16
        num_key_value_heads = 4
        num_hidden_layers = 2
        hidden_size = 256
        head_dim = 64
        diversity_enforce = False
        epsilon_greedy = False
        epsilon = 0.0
        diversity_enforce_barrier = 0

    return FakeConfig()


def make_test_sequences(batch_size, block_length, mask_token_id, vocab_size,
                        denoising_step=0, denoising_steps=4, device="cuda"):
    """Create test sequences in DENOISING state with random intermediate tokens."""
    seqs = []
    for i in range(batch_size):
        # Simple prompt
        prompt = list(range(1, block_length * 2 + 1))
        from jetengine.sampling_params import SamplingParams
        sp = SamplingParams(
            max_tokens=256,
            block_length=block_length,
            denoising_steps=denoising_steps,
            temperature=1.0,
            topk=50,
            topp=1.0,
            remasking_strategy="low_confidence_dynamic",
            dynamic_threshold=0.9,
        )
        seq = Sequence(prompt, mask_token_id, sp)
        # Force to DENOISING state (simulate after prefill)
        seq.status = SequenceStatus.DENOISING
        seq.num_cached_tokens = len(prompt)
        seq.current_denoising_step = denoising_step
        seq.global_denoising_step = denoising_step

        # Create intermediate block with some masks
        block = torch.randint(0, vocab_size, (block_length,), dtype=torch.long, device=device)
        # Put some mask tokens
        num_masks = max(1, block_length - denoising_step)
        mask_positions = torch.randperm(block_length)[:num_masks]
        block[mask_positions] = mask_token_id
        seq.intermediate_block_tokens = block
        seq.intermediate_block_tokens_entropy = torch.zeros(block_length, dtype=torch.float, device=device)
        seq.block_trajectory = torch.zeros(block_length, dtype=torch.long, device=device)
        seq.block_logprobs = torch.zeros(block_length, dtype=torch.float, device=device)
        seq.block_entropies = torch.zeros(block_length, dtype=torch.float, device=device)
        seq.block_table = list(range(i * 2, i * 2 + 2))  # fake block table

        seqs.append(seq)
    return seqs


def deep_copy_seqs(seqs):
    """Deep copy sequence state for comparison.
    Note: Sequence defines __getstate__/__setstate__ which drops many attrs,
    so we must copy __dict__ directly instead of using copy.copy."""
    copies = []
    for seq in seqs:
        c = Sequence.__new__(Sequence)
        # Copy all attributes from __dict__
        c.__dict__.update(seq.__dict__)
        # Deep copy mutable fields
        c.token_ids = list(seq.token_ids)
        c.intermediate_block_tokens = seq.intermediate_block_tokens.clone()
        c.intermediate_block_tokens_entropy = seq.intermediate_block_tokens_entropy.clone()
        c.block_trajectory = seq.block_trajectory.clone()
        c.block_logprobs = seq.block_logprobs.clone()
        c.block_entropies = seq.block_entropies.clone()
        c.block_table = list(seq.block_table)
        c.trajectory = list(seq.trajectory)
        c.logprobs = list(seq.logprobs)
        c.entropies = list(seq.entropies)
        c.token_counts = Counter(seq.token_counts)
        c.num_transfer_tokens_per_step = list(seq.num_transfer_tokens_per_step)
        c.stop_words = set(seq.stop_words)
        copies.append(c)
    return copies


def test_postprocess_unify_vs_loop():
    """Test that postprocess_unify gives same results as postprocess_loop for DENOISING."""
    print("Test: postprocess_unify vs postprocess_loop consistency")

    config = make_test_config()
    vocab_size = config.hf_config.vocab_size
    block_length = config.block_length
    mask_token_id = config.mask_token_id
    batch_size = 8
    device = "cuda"

    # Set deterministic seed
    torch.manual_seed(42)

    # Create random logits (simulating model output)
    logits = torch.randn(batch_size * block_length, vocab_size, device=device, dtype=torch.float32)

    for denoising_step in range(4):
        # Create two identical sets of sequences
        seqs_unify = make_test_sequences(
            batch_size, block_length, mask_token_id, vocab_size,
            denoising_step=denoising_step, device=device
        )
        seqs_loop = deep_copy_seqs(seqs_unify)

        # Create two schedulers
        sched_unify = Scheduler(config)
        sched_unify.consistent_sampling_params = True

        sched_loop = Scheduler(config)
        sched_loop.consistent_sampling_params = True

        # Both use same seed for sampling
        torch.manual_seed(123 + denoising_step)
        sched_unify.postprocess_unify(seqs_unify, logits.clone(), RunType.DENOISE)

        torch.manual_seed(123 + denoising_step)
        sched_loop.postprocess_loop(seqs_loop, logits.clone(), RunType.DENOISE)

        # Compare results
        all_match = True
        for i in range(batch_size):
            su, sl = seqs_unify[i], seqs_loop[i]

            # Compare status
            if su.status != sl.status:
                print(f"  FAIL step={denoising_step} seq={i}: status {su.status} vs {sl.status}")
                all_match = False

            # Compare intermediate tokens
            if not torch.equal(su.intermediate_block_tokens, sl.intermediate_block_tokens):
                print(f"  FAIL step={denoising_step} seq={i}: intermediate_block_tokens differ")
                print(f"    unify: {su.intermediate_block_tokens}")
                print(f"    loop:  {sl.intermediate_block_tokens}")
                all_match = False

            # Compare denoising step
            if su.current_denoising_step != sl.current_denoising_step:
                print(f"  FAIL step={denoising_step} seq={i}: denoising_step {su.current_denoising_step} vs {sl.current_denoising_step}")
                all_match = False

        if all_match:
            print(f"  PASS denoising_step={denoising_step}")
        else:
            print(f"  FAIL denoising_step={denoising_step}")


def test_saving_commit_correctness():
    """Test that SAVING sequences correctly commit blocks and transition state."""
    print("\nTest: SAVING commit correctness")

    config = make_test_config()
    block_length = config.block_length
    mask_token_id = config.mask_token_id
    device = "cuda"

    torch.manual_seed(42)

    # Create sequences in SAVING state (fully denoised)
    seqs = make_test_sequences(4, block_length, mask_token_id, config.hf_config.vocab_size, device=device)
    for seq in seqs:
        seq.status = SequenceStatus.SAVING
        # Replace all mask tokens with real tokens (fully denoised)
        seq.intermediate_block_tokens = torch.randint(
            1, 1000, (block_length,), dtype=torch.long, device=device
        )
        seq.current_denoising_step = 4

    # Record pre-commit state
    pre_num_tokens = [seq.num_tokens for seq in seqs]
    pre_block_tokens = [seq.intermediate_block_tokens.clone() for seq in seqs]

    # Create dummy logits (not used for SAVING)
    logits = torch.randn(len(seqs) * block_length, config.hf_config.vocab_size, device=device)

    sched = Scheduler(config)
    sched.consistent_sampling_params = True
    sched.running = list(seqs)  # Need to set running for finished seq filtering
    sched.postprocess_unify(seqs, logits, RunType.DENOISE)

    all_pass = True
    for i, seq in enumerate(seqs):
        # Tokens should have been extended
        expected_len = pre_num_tokens[i] + block_length
        if seq.num_tokens != expected_len:
            print(f"  FAIL seq={i}: expected {expected_len} tokens, got {seq.num_tokens}")
            all_pass = False

        # The committed tokens should match pre_block_tokens
        committed = seq.token_ids[-block_length:]
        expected = pre_block_tokens[i].tolist()
        if committed != expected:
            print(f"  FAIL seq={i}: committed tokens don't match")
            all_pass = False

        # Status should be DENOISING (started new block) or FINISHED
        if seq.status not in (SequenceStatus.DENOISING, SequenceStatus.FINISHED):
            print(f"  FAIL seq={i}: unexpected status {seq.status}")
            all_pass = False

    if all_pass:
        print("  PASS")


def test_entropy_computation():
    """Test that entropy values are non-negative and reasonable."""
    print("\nTest: entropy computation sanity")

    device = "cuda"
    vocab_size = 1000
    batch_size = 4
    block_length = 4

    # Create uniform-ish logits
    logits = torch.randn(batch_size * block_length, vocab_size, device=device)
    probs = torch.softmax(logits, dim=-1).view(batch_size, block_length, -1)

    EPS = 1e-12
    log_probs = probs.clamp_min(EPS).log()
    entropies = -(probs * log_probs).sum(dim=-1)

    # Entropy should be non-negative
    assert (entropies >= -1e-6).all(), f"Negative entropy found: {entropies.min()}"

    # For random logits over 1000 classes, entropy should be substantial
    assert entropies.mean() > 1.0, f"Entropy too low: {entropies.mean()}"

    # Max entropy for uniform distribution = log(vocab_size)
    max_ent = torch.log(torch.tensor(float(vocab_size)))
    assert (entropies <= max_ent + 0.01).all(), f"Entropy exceeds max: {entropies.max()} > {max_ent}"

    print(f"  PASS (mean entropy={entropies.mean():.3f}, max_possible={max_ent:.3f})")


def test_batch_commit_from_cpu():
    """Test _commit_block_from_cpu matches commit_block."""
    print("\nTest: _commit_block_from_cpu vs commit_block")

    block_length = 4
    mask_token_id = 151669
    from jetengine.sampling_params import SamplingParams
    sp = SamplingParams(max_tokens=256, block_length=block_length, denoising_steps=4)

    torch.manual_seed(42)
    block_tokens = torch.randint(1, 1000, (block_length,), dtype=torch.long)
    traj = torch.arange(1, block_length + 1, dtype=torch.long)
    logp = torch.randn(block_length)
    ent = torch.rand(block_length)

    # Method 1: commit_block (original)
    prompt = list(range(1, block_length * 2 + 1))
    seq1 = Sequence(prompt, mask_token_id, sp)
    seq1.status = SequenceStatus.SAVING
    seq1.intermediate_block_tokens = block_tokens.clone()
    seq1.block_trajectory = traj.clone()
    seq1.block_logprobs = logp.clone()
    seq1.block_entropies = ent.clone()
    seq1.current_denoising_step = 4
    seq1.num_cached_tokens = len(prompt)
    seq1.commit_block(block_tokens.clone())

    # Method 2: _commit_block_from_cpu
    seq2 = Sequence(prompt, mask_token_id, sp)
    seq2.status = SequenceStatus.SAVING
    seq2.intermediate_block_tokens = block_tokens.clone()
    seq2.block_trajectory = traj.clone()
    seq2.block_logprobs = logp.clone()
    seq2.block_entropies = ent.clone()
    seq2.current_denoising_step = 4
    seq2.num_cached_tokens = len(prompt)
    seq2._commit_block_from_cpu(
        block_tokens.tolist(), traj.tolist(), logp.tolist(), ent.tolist()
    )

    assert seq1.token_ids == seq2.token_ids, f"token_ids differ: {seq1.token_ids} vs {seq2.token_ids}"
    assert seq1.num_tokens == seq2.num_tokens, f"num_tokens differ"
    assert seq1.status == seq2.status, f"status differ: {seq1.status} vs {seq2.status}"
    assert seq1.trajectory == seq2.trajectory, f"trajectory differ"
    # logprobs/entropies might differ in float precision, check close
    for a, b in zip(seq1.logprobs, seq2.logprobs):
        assert abs(a - b) < 1e-5, f"logprobs differ: {a} vs {b}"

    print("  PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        # Run CPU-only tests
        test_batch_commit_from_cpu()
        print("\nCPU tests passed.")
        sys.exit(0)

    test_entropy_computation()
    test_batch_commit_from_cpu()
    test_saving_commit_correctness()
    test_postprocess_unify_vs_loop()
    print("\nAll tests completed.")
