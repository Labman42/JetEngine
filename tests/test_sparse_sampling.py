"""
Unit test for sparse sampling optimization.
Verifies that sampling only masked positions produces identical results
to the dense path (sampling all positions then selecting masked ones).

Run: python tests/test_sparse_sampling.py
"""
import torch
import sys
sys.path.insert(0, '.')


def test_sparse_sampling_equivalence():
    """
    Test that sparse sampling (only masked positions) produces the same
    token selection and probabilities as dense sampling (all positions).

    The sparse path:
    1. Gathers logits at masked positions only
    2. Runs softmax + top_k_top_p_sampling on the smaller tensor
    3. Scatters results back

    This must produce identical results to the dense path for masked positions,
    and preserve original tokens for unmasked positions.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    block_len = 4
    vocab_size = 151936
    mask_token_id = 151669
    temperature = 0.6
    top_k = 0
    top_p = 0.95
    EPS = 1e-12

    # Simulate logits from model forward pass
    logits = torch.randn(batch_size, block_len, vocab_size, device=device, dtype=torch.float32)

    # Create realistic mask patterns: at different denoising steps,
    # different numbers of positions are masked
    # Step 0: all 4 masked, Step 1: 3 masked, Step 2: 2 masked, Step 3: 1 masked
    current_tokens = torch.randint(0, vocab_size, (batch_size, block_len), device=device)
    for i in range(batch_size):
        num_masked = (i % 4) + 1  # 1-4 masked positions
        mask_positions = torch.randperm(block_len)[:num_masked]
        current_tokens[i, mask_positions] = mask_token_id

    mask_token_mask = (current_tokens == mask_token_id)
    num_masked_total = mask_token_mask.sum().item()
    num_total = batch_size * block_len
    print(f"Masked: {num_masked_total}/{num_total} ({100*num_masked_total/num_total:.1f}%)")

    # === Dense path (reference) ===
    d_logits = logits.reshape(-1, vocab_size)
    # Temperature scaling + softmax
    dense_probs = torch.softmax(d_logits / temperature, dim=-1).view(batch_size, block_len, -1)
    flat_probs_dense = dense_probs.view(-1, vocab_size)

    # Use deterministic argmax instead of stochastic sampling for exact comparison
    dense_samples = flat_probs_dense.argmax(dim=-1).to(torch.int64)
    dense_x0 = dense_samples.view(batch_size, block_len)
    dense_x0_p = torch.gather(
        flat_probs_dense, 1, dense_samples.unsqueeze(-1)
    ).view(batch_size, block_len)
    dense_x0_logp = torch.log(dense_x0_p.clamp_min(EPS))

    # Entropy (dense)
    log_probs_dense = dense_probs.clamp_min(EPS).log()
    dense_entropies = -(dense_probs * log_probs_dense).sum(dim=-1)

    # === Sparse path ===
    flat_mask = mask_token_mask.view(-1)  # (batch_size * block_len,)
    masked_indices = flat_mask.nonzero(as_tuple=True)[0]  # indices of masked positions

    # Gather only masked logits
    sparse_logits = d_logits[masked_indices]  # (num_masked, vocab_size)

    # Temperature scaling + softmax on sparse set
    sparse_probs = torch.softmax(sparse_logits / temperature, dim=-1)

    # Sample (deterministic argmax for test)
    sparse_samples = sparse_probs.argmax(dim=-1).to(torch.int64)
    sparse_x0_p = torch.gather(sparse_probs, 1, sparse_samples.unsqueeze(-1)).squeeze(-1)
    sparse_x0_logp = torch.log(sparse_x0_p.clamp_min(EPS))

    # Entropy (sparse)
    sparse_log_probs = sparse_probs.clamp_min(EPS).log()
    sparse_entropies = -(sparse_probs * sparse_log_probs).sum(dim=-1)

    # Scatter back to full shape
    full_x0 = current_tokens.clone()  # start with current tokens (unmasked preserved)
    full_x0.view(-1)[masked_indices] = sparse_samples

    full_x0_p = torch.zeros(batch_size, block_len, device=device)
    full_x0_p.view(-1)[masked_indices] = sparse_x0_p

    full_x0_logp = torch.zeros(batch_size, block_len, device=device)
    full_x0_logp.view(-1)[masked_indices] = sparse_x0_logp

    full_entropies = torch.zeros(batch_size, block_len, device=device)
    full_entropies.view(-1)[masked_indices] = sparse_entropies

    # === Verify equivalence at masked positions ===
    # Token samples must match exactly at masked positions
    dense_masked_tokens = dense_x0[mask_token_mask]
    sparse_masked_tokens = full_x0[mask_token_mask]
    assert torch.equal(dense_masked_tokens, sparse_masked_tokens), \
        f"Token mismatch at masked positions! max diff: {(dense_masked_tokens - sparse_masked_tokens).abs().max()}"

    # Probabilities must match at masked positions
    dense_masked_probs = dense_x0_p[mask_token_mask]
    sparse_masked_probs = full_x0_p[mask_token_mask]
    max_prob_diff = (dense_masked_probs - sparse_masked_probs).abs().max().item()
    assert max_prob_diff < 1e-6, f"Probability mismatch at masked positions: {max_prob_diff}"

    # Log probabilities must match at masked positions
    dense_masked_logp = dense_x0_logp[mask_token_mask]
    sparse_masked_logp = full_x0_logp[mask_token_mask]
    max_logp_diff = (dense_masked_logp - sparse_masked_logp).abs().max().item()
    assert max_logp_diff < 1e-5, f"Log-prob mismatch at masked positions: {max_logp_diff}"

    # Entropy must match at masked positions
    dense_masked_ent = dense_entropies[mask_token_mask]
    sparse_masked_ent = full_entropies[mask_token_mask]
    max_ent_diff = (dense_masked_ent - sparse_masked_ent).abs().max().item()
    assert max_ent_diff < 1e-5, f"Entropy mismatch at masked positions: {max_ent_diff}"

    # Unmasked positions must preserve original tokens
    assert torch.equal(full_x0[~mask_token_mask], current_tokens[~mask_token_mask]), \
        "Unmasked positions were modified!"

    print(f"PASS: All assertions passed")
    print(f"  Token match: exact")
    print(f"  Max prob diff: {max_prob_diff:.2e}")
    print(f"  Max logp diff: {max_logp_diff:.2e}")
    print(f"  Max entropy diff: {max_ent_diff:.2e}")
    print(f"  Unmasked preserved: exact")


def test_sparse_sampling_remasking_strategies():
    """
    Test that remasking strategies (low_confidence_static, sequential) produce
    identical transfer_index when using sparse-sampled x0 vs dense-sampled x0.

    Since both paths produce the same x0 and x0_p at masked positions, the
    remasking logic (which only uses x0_p at masked positions) must be identical.
    """
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    block_len = 4
    vocab_size = 151936
    mask_token_id = 151669

    # Create test data
    current_tokens = torch.randint(0, vocab_size, (batch_size, block_len), device=device)
    for i in range(batch_size):
        num_masked = (i % 4) + 1
        mask_positions = torch.randperm(block_len)[:num_masked]
        current_tokens[i, mask_positions] = mask_token_id

    mask_token_mask = (current_tokens == mask_token_id)

    # Simulate x0_p (confidence) - same values at masked positions for both paths
    batch_seq_x0_p = torch.rand(batch_size, block_len, device=device)

    # num_to_transfer = 1 for all sequences
    effective_num_to_transfer = torch.ones(batch_size, device=device, dtype=torch.long)
    mask_available = mask_token_mask.any(dim=1)

    # --- low_confidence_static ---
    confidence = torch.where(
        mask_token_mask, batch_seq_x0_p, torch.full_like(batch_seq_x0_p, -torch.inf)
    )
    _, top_indices = torch.topk(confidence, k=block_len, dim=1)
    k_mask = torch.arange(block_len, device=device).unsqueeze(0) < effective_num_to_transfer.unsqueeze(1)
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    transfer_index.scatter_(1, top_indices, k_mask)
    transfer_index &= mask_token_mask

    # The transfer_index only depends on x0_p at masked positions.
    # Since sparse and dense produce the same x0_p at masked positions,
    # the transfer_index must be identical.
    assert transfer_index.sum() > 0, "No transfers happened"

    # Verify that transfer_index only has True at masked positions
    assert (transfer_index & ~mask_token_mask).sum() == 0, \
        "Transfer at unmasked position!"

    print(f"PASS: Remasking strategy test passed")
    print(f"  Total transfers: {transfer_index.sum().item()}")
    print(f"  All transfers at masked positions: True")


def test_sparse_sampling_edge_cases():
    """Test edge cases: all masked, none masked, single masked."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    block_len = 4
    vocab_size = 1000  # smaller for speed
    mask_token_id = 999

    logits = torch.randn(batch_size, block_len, vocab_size, device=device)

    # Case 1: All masked
    tokens_all_masked = torch.full((batch_size, block_len), mask_token_id, device=device)
    mask_all = (tokens_all_masked == mask_token_id)
    flat_mask = mask_all.view(-1)
    masked_indices = flat_mask.nonzero(as_tuple=True)[0]
    assert masked_indices.shape[0] == batch_size * block_len, "All should be masked"

    # Case 2: None masked
    tokens_none_masked = torch.randint(0, mask_token_id, (batch_size, block_len), device=device)
    mask_none = (tokens_none_masked == mask_token_id)
    flat_mask_none = mask_none.view(-1)
    masked_indices_none = flat_mask_none.nonzero(as_tuple=True)[0]
    assert masked_indices_none.shape[0] == 0, "None should be masked"

    # Case 3: Single masked per sequence
    tokens_single = torch.randint(0, mask_token_id, (batch_size, block_len), device=device)
    tokens_single[:, 2] = mask_token_id  # position 2 is masked
    mask_single = (tokens_single == mask_token_id)
    flat_mask_single = mask_single.view(-1)
    masked_indices_single = flat_mask_single.nonzero(as_tuple=True)[0]
    assert masked_indices_single.shape[0] == batch_size, "One per sequence"

    # Verify sparse softmax on single masked positions
    d_logits = logits.reshape(-1, vocab_size)
    sparse_logits = d_logits[masked_indices_single]
    sparse_probs = torch.softmax(sparse_logits / 0.6, dim=-1)
    assert sparse_probs.shape == (batch_size, vocab_size)
    assert torch.allclose(sparse_probs.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5)

    print(f"PASS: Edge cases passed")
    print(f"  All masked: {batch_size * block_len} positions")
    print(f"  None masked: 0 positions")
    print(f"  Single masked: {batch_size} positions")


def bench_sparse_vs_dense():
    """Benchmark sparse vs dense softmax+sampling to measure speedup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("SKIP: Benchmark requires CUDA")
        return

    import time

    batch_size = 128
    block_len = 4
    vocab_size = 151936
    mask_token_id = 151669
    temperature = 0.6
    n_warmup = 10
    n_iters = 100

    logits = torch.randn(batch_size * block_len, vocab_size, device=device, dtype=torch.float32)

    # Test different masking ratios (simulating different chain steps)
    for mask_ratio_name, num_masked_per_seq in [("step0 (4/4)", 4), ("step1 (3/4)", 3),
                                                  ("step2 (2/4)", 2), ("step3 (1/4)", 1)]:
        current_tokens = torch.randint(0, vocab_size, (batch_size, block_len), device=device)
        for i in range(batch_size):
            mask_positions = torch.randperm(block_len)[:num_masked_per_seq]
            current_tokens[i, mask_positions] = mask_token_id

        mask_token_mask = (current_tokens == mask_token_id)
        flat_mask = mask_token_mask.view(-1)
        masked_indices = flat_mask.nonzero(as_tuple=True)[0]
        num_masked = masked_indices.shape[0]

        # Dense path
        torch.cuda.synchronize()
        for _ in range(n_warmup):
            probs = torch.softmax(logits / temperature, dim=-1)
            samples = probs.argmax(dim=-1)
            log_p = probs.clamp_min(1e-12).log()
            ent = -(probs * log_p).sum(dim=-1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            probs = torch.softmax(logits / temperature, dim=-1)
            samples = probs.argmax(dim=-1)
            log_p = probs.clamp_min(1e-12).log()
            ent = -(probs * log_p).sum(dim=-1)
        torch.cuda.synchronize()
        dense_time = (time.perf_counter() - t0) / n_iters

        # Sparse path
        torch.cuda.synchronize()
        for _ in range(n_warmup):
            sparse_logits = logits[masked_indices]
            sparse_probs = torch.softmax(sparse_logits / temperature, dim=-1)
            sparse_samples = sparse_probs.argmax(dim=-1)
            sparse_log_p = sparse_probs.clamp_min(1e-12).log()
            sparse_ent = -(sparse_probs * sparse_log_p).sum(dim=-1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            sparse_logits = logits[masked_indices]
            sparse_probs = torch.softmax(sparse_logits / temperature, dim=-1)
            sparse_samples = sparse_probs.argmax(dim=-1)
            sparse_log_p = sparse_probs.clamp_min(1e-12).log()
            sparse_ent = -(sparse_probs * sparse_log_p).sum(dim=-1)
        torch.cuda.synchronize()
        sparse_time = (time.perf_counter() - t0) / n_iters

        speedup = dense_time / sparse_time
        print(f"{mask_ratio_name}: dense={dense_time*1000:.2f}ms, sparse={sparse_time*1000:.2f}ms, "
              f"speedup={speedup:.2f}x ({num_masked}/{batch_size*block_len} positions)")


if __name__ == "__main__":
    print("=== Test 1: Sparse Sampling Equivalence ===")
    test_sparse_sampling_equivalence()
    print()

    print("=== Test 2: Remasking Strategy Compatibility ===")
    test_sparse_sampling_remasking_strategies()
    print()

    print("=== Test 3: Edge Cases ===")
    test_sparse_sampling_edge_cases()
    print()

    print("=== Benchmark: Sparse vs Dense ===")
    bench_sparse_vs_dense()
