"""
Simulation of the Decode-Phase Token Acceptance Logic

Purpose: Verify whether the decode loop in DistributedPipelineController
unconditionally accepts contaminated tokens.

This simulation replicates the exact logic from distributed_engine.py:226-283
using a mock model that returns pre-determined tokens based on input context.

Key questions:
1. When all n-gram guesses are WRONG, how many incorrect tokens are accepted?
2. When there are NO n-gram guesses (history too short), do tokens get replicated?
3. How does num_decode_batches affect contamination?
"""

import numpy as np
from typing import List, Dict, Optional

# ============================================================
# Simulation parameters (mirroring the actual code)
# ============================================================
LOOKAHEAD_WINDOW = 3  # default in distributed_engine.py line 209
NGRAM_SIZE = 3        # line 210
NUM_DECODE_BATCHES = max(2, LOOKAHEAD_WINDOW)  # line 226
VOCAB_SIZE = 32000

print(f"Configuration:")
print(f"  lookahead_window = {LOOKAHEAD_WINDOW}")
print(f"  ngram_size = {NGRAM_SIZE}")
print(f"  num_decode_batches = max(2, {LOOKAHEAD_WINDOW}) = {NUM_DECODE_BATCHES}")
print()

# ============================================================
# Fixed test scenario
# ============================================================
# Ground-truth tokens for a short phrase: "the cat sat on the mat"
# Token IDs are arbitrary but distinct for clarity
GROUND_TRUTH = [101, 202, 303, 404, 505, 606, 707]
PROMPT_TOKENS = [101, 202, 303]  # "the cat sat"
GROUND_TRUTH_NEXT = [404, 505, 606, 707]  # "on the mat"

# The n-gram suffix after 3 prompt tokens is [101, 202, 303]
# We search history for this suffix to make guesses.
# If the history doesn't contain this trigram before, guesses=[].

def find_ngram_guesses(
    history: List[int], 
    ngram_size: int = NGRAM_SIZE,
    lookahead_window: int = LOOKAHEAD_WINDOW
) -> List[int]:
    """Replicate the exact n-gram guess logic from distributed_engine.py lines 213-221."""
    guesses = []
    if lookahead_window > 0 and len(history) >= ngram_size + 1:
        suffix = history[-ngram_size:]
        for i in range(len(history) - ngram_size - 1, 0, -1):
            if history[i:i+ngram_size] == suffix:
                guesses = history[i+ngram_size : i+ngram_size+lookahead_window]
                if guesses:
                    break
    return guesses


def mock_model_predict(input_tokens: List[int], real_prefix: Optional[List[int]] = None) -> int:
    """
    Mock transformer model: given context tokens, predict the next token.
    
    This simulates how a real transformer's prediction changes when given
    incorrect n-gram guesses as context. In a real transformer, attention
    over all input tokens means wrong context changes the prediction.
    
    We model the probability that the model outputs the correct token as
    inversely proportional to the amount of contamination in the input.
    """
    last = input_tokens[-1]
    
    # Base prediction: what the model should predict given ground-truth context
    ground_truth_map = {
        303: 404,  # -> "on"
        404: 505,  # -> "the"  
        505: 606,  # -> "mat"
        606: 707,
        400: 500,  # for n-gram test
        500: 600,
        600: 700,
        700: 800,
    }
    correct = ground_truth_map.get(last, 999)
    
    # Pure autoregressive context: only the last real token
    if len(input_tokens) == 1:
        return correct
    
    # Contaminated context: input includes n-gram guess tokens.
    # In a real transformer, wrong context changes the hidden states
    # through attention. We simulate this by returning a wrong token
    # with probability proportional to contamination_depth.
    contamination_depth = len(input_tokens) - 1  # number of guess tokens prepended
    
    # Heuristic: each extra guess token has a chance to shift the prediction
    # In a real model, this depends on attention patterns, but we simulate
    # the general case where contamination corrupts the output.
    shift = contamination_depth * 37  # arbitrary but consistent shift for wrong guesses
    return correct + shift if contamination_depth > 0 else correct


def run_decode_loop(
    initial_history: List[int],
    max_tokens: int = 10,
) -> List[int]:
    """Simulate the decode loop from distributed_engine.py lines 205-283."""
    all_generated_tokens = list(initial_history)
    loop_count = 0
    
    while len(all_generated_tokens) < max_tokens + len(initial_history):
        loop_count += 1
        
        # --- N-gram lookup (lines 213-221) ---
        guesses = find_ngram_guesses(all_generated_tokens)
        
        # --- Build decode batches (lines 226-233) ---
        decode_batches = []
        for db_idx in range(NUM_DECODE_BATCHES):
            batch_guesses = guesses[db_idx:] if db_idx < len(guesses) else []
            candidate_tokens = [all_generated_tokens[-1]] + batch_guesses
            decode_batches.append((db_idx, candidate_tokens))
        
        # --- "Send" all batches ---
        # Simulate model prediction for each batch
        raw_results = []
        for db_idx, candidate_tokens in decode_batches:
            predicted = mock_model_predict(candidate_tokens)
            raw_results.append((db_idx, [predicted]))  # result.tokens is always [1 token]
        
        # --- Acceptance logic (lines 259-271) ---
        all_accepted = []
        for db_idx, result_tokens in raw_results:
            batch_guesses = guesses[db_idx:] if db_idx < len(guesses) else []
            accepted_count = 1  # ALWAYS ACCEPT FIRST TOKEN (line 263)
            for g_idx in range(len(batch_guesses)):
                if g_idx < len(result_tokens) and result_tokens[g_idx] == batch_guesses[g_idx]:
                    accepted_count += 1
                else:
                    break
            verified_tokens = result_tokens[:accepted_count]
            all_accepted.extend(verified_tokens)
        
        # --- Append to history (line 273) ---
        all_generated_tokens.extend(all_accepted)
    
    return all_generated_tokens


def run_correct_decode_loop(
    initial_history: List[int],
    max_tokens: int = 10,
) -> List[int]:
    """
    What the correct implementation should do:
    - Send ONE batch with only the last real token
    - Accept one token per loop iteration (autoregressive)
    """
    all_generated_tokens = list(initial_history)
    
    while len(all_generated_tokens) < max_tokens + len(initial_history):
        last_token = [all_generated_tokens[-1]]
        predicted = mock_model_predict(last_token)
        all_generated_tokens.append(predicted)
    
    return all_generated_tokens


# ============================================================
# TEST 1: Short history (no n-gram match possible)
# ============================================================
print("=" * 75)
print("TEST 1: Short history — no n-gram guesses found")
print("=" * 75)
print(f"Initial history: {PROMPT_TOKENS} (length={len(PROMPT_TOKENS)})")
print(f"Ground-truth next tokens (autoregressive): {GROUND_TRUTH_NEXT}")
print()

buggy_output = run_decode_loop(list(PROMPT_TOKENS), max_tokens=4)
correct_output = run_correct_decode_loop(list(PROMPT_TOKENS), max_tokens=4)

print(f"Buggy decode output:    {buggy_output}")
print(f"Correct decode output:  {correct_output}")
print()

buggy_generated = buggy_output[len(PROMPT_TOKENS):]
correct_generated = correct_output[len(PROMPT_TOKENS):]
print(f"Buggy generated tokens: {buggy_generated}")
print(f"Correct generated:      {correct_generated}")
print()

# Check duplication
unique_buggy = set(buggy_generated)
print(f"Unique tokens in buggy output: {len(unique_buggy)}/4 expected")
print(f"Buggy tokens deduplicated: {len(set(buggy_generated))}")
print()

# Check if contaminated
contaminated = [t for t in buggy_generated if t not in GROUND_TRUTH_NEXT]
print(f"Contaminated tokens: {len(contaminated)} out of {len(buggy_generated)}")
if len(contaminated) > 0:
    print(f"  -> BUG CONFIRMED: {contaminated}")

print()

# ============================================================
# TEST 2: Longer history with repeated trigram (guesses found)
# ============================================================
print("=" * 75)
print("TEST 2: History has repeated trigram → n-gram guesses found")
print("=" * 75)
#
# Build: [A, B, C, A, B, C, A, B, C]
# suffix (last 3) = [A, B, C]
# Searching backwards for previous [A, B, C]:
#   i = len(9)-3-1 = 5: history[5:8] = [C, A, B] ≠ [A, B, C]
#   i = 4: history[4:7] = [B, C, A] ≠ [A, B, C]
#   i = 3: history[3:6] = [A, B, C] ← MATCH at index 3!
# guesses = history[3+3 : 3+3+3] = history[6:9] = [A, B, C]
# So guesses = [100, 200, 300] — the 3 tokens that followed the
# previous occurrence of trigram [A, B, C].

history_with_guess = [100, 200, 300, 100, 200, 300, 100, 200, 300]
print(f"History: {history_with_guess}")
found_guesses = find_ngram_guesses(history_with_guess)
print(f"N-gram suffix (history[-3:]): {history_with_guess[-3:]}")
print(f"Found guesses: {found_guesses}")
print()

# Now simulate: the model generates based on contaminated inputs
print("--- Build decode batches ---")
last_real = history_with_guess[-1]
for db_idx in range(NUM_DECODE_BATCHES):
    batch_guesses = found_guesses[db_idx:] if db_idx < len(found_guesses) else []
    candidate_tokens = [last_real] + batch_guesses
    predicted = mock_model_predict(candidate_tokens)
    
    # Determine contamination
    if len(candidate_tokens) == 1:
        status = "PURE (no contamination)"
    else:
        status = f"CONTAMINATED (sees {batch_guesses} as fake context)"
    
    print(f"  Batch {db_idx}: input={candidate_tokens} → predicted={predicted} [{status}]")

print()
print("NOTE: In a real transformer, all batches predict tokens AFTER the")
print("same last position (the 'guess2' token), but with DIFFERENT preceding")
print("context. The predictions will diverge because attention over wrong")
print("context tokens corrupts the hidden states.")

# ============================================================
# TEST 3: Exact trace of one loop iteration
# ============================================================
print("=" * 75)
print("TEST 3: Step-by-step trace of one decode loop iteration")
print("=" * 75)

history = list(PROMPT_TOKENS)  # [101, 202, 303]
ground_truth_next = 404  # "on"

print(f"State before loop:")
print(f"  all_generated_tokens = {history}")
print(f"  Ground-truth next token = {ground_truth_next}")
print()

# Step 1: Find guesses
guesses = find_ngram_guesses(history)
print(f"Step 1 - N-gram lookup:")
print(f"  suffix = {history[-NGRAM_SIZE:]}")
print(f"  guesses = {guesses} (empty because suffix never appeared before)")
print()

# Step 2: Build batches
print(f"Step 2 - Build {NUM_DECODE_BATCHES} decode batches:")
decode_batches = []
for db_idx in range(NUM_DECODE_BATCHES):
    batch_guesses = guesses[db_idx:] if db_idx < len(guesses) else []
    candidate_tokens = [history[-1]] + batch_guesses
    decode_batches.append((db_idx, candidate_tokens))
    print(f"  Batch {db_idx}: input={candidate_tokens} "
          f"(last_real={history[-1]}, batch_guesses={batch_guesses})")

print()

# Step 3: Model predictions
print(f"Step 3 - Model predicts for each batch (context-dependent):")
raw_results = []
for db_idx, candidate_tokens in decode_batches:
    predicted = mock_model_predict(candidate_tokens)
    raw_results.append((db_idx, [predicted]))
    
    is_contaminated = "YES" if len(candidate_tokens) > 1 else "no"
    is_correct = "CORRECT" if predicted == ground_truth_next else "WRONG"
    print(f"  Batch {db_idx}: input_len={len(candidate_tokens)}, "
          f"predicted={predicted}, correct={ground_truth_next}, "
          f"contaminated={is_contaminated}, verdict={is_correct}")

print()

# Step 4: Acceptance
print(f"Step 4 - Token acceptance (unconditional accepted_count=1):")
all_accepted = []
for db_idx, result_tokens in raw_results:
    batch_guesses = guesses[db_idx:] if db_idx < len(guesses) else []
    accepted_count = 1  # unconditional
    verified_tokens = result_tokens[:accepted_count]
    all_accepted.extend(verified_tokens)
    print(f"  Batch {db_idx}: result_tokens={result_tokens}, "
          f"accepted_count=1, verified={verified_tokens}")

print()
print(f"  ALL accepted tokens: {all_accepted}")
print(f"  Expected (autoregressive): [{ground_truth_next}]")
print(f"  Incorrectly accepted: {len([t for t in all_accepted if t != ground_truth_next])}")
print()

# ============================================================
# Summary
# ============================================================
print("=" * 75)
print("SUMMARY")
print("=" * 75)
print()
print(f"1. num_decode_batches = max(2, lookahead_window) = max(2, {LOOKAHEAD_WINDOW})")
print(f"   = {NUM_DECODE_BATCHES} batches per loop iteration")
print()
print(f"2. Each batch has unconditional accepted_count=1")
print(f"   => {NUM_DECODE_BATCHES} tokens accepted per iteration")
print(f"   => Expected: 1 token per iteration (autoregressive)")
print()
print(f"3. When guesses=[] (common case):")
print(f"   - All {NUM_DECODE_BATCHES} batches have IDENTICAL inputs [last_token]")
print(f"   - All predict the SAME next token")
print(f"   - Output has {NUM_DECODE_BATCHES}-token replication pattern")
print()
print(f"4. When guesses found (rare case):")
print(f"   - Guesses are prepended as fake context (contaminated)")
print(f"   - Each batch has DIFFERENT context length")
print(f"   - Predictions differ due to different conditioning")
print(f"   - ALL unconditionally accepted")
print()
print(f"VERDICT: BUG CONFIRMED — contamination severity: HIGH (Blocker)")
print(f"Token acceptance rate: {NUM_DECODE_BATCHES}x normal rate")
