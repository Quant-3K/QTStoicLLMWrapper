# API Reference — QTStoic CC v2.1.3

Complete reference of all classes and functions as implemented in the codebase.

**Terminology:**
- **CC** — Coupling Constraint (the two-channel admissibility gate)
- **KQ** — Coherence Quality (the composite metric measuring generation coherence)

---

## Table of Contents

1. [Classes](#1-classes)
2. [Sensor Functions](#2-sensor-functions)
3. [JS Divergence Functions](#3-js-divergence-functions)
4. [Coherence Quality (KQ) Functions](#4-coherence-quality-kq-functions)
5. [CC Gate Functions](#5-cc-gate-functions)
6. [Lambda Functions](#6-lambda-functions)
7. [Harm Channel Functions](#7-harm-channel-functions)
8. [Structural Channel Functions](#8-structural-channel-functions)
9. [State Management Functions](#9-state-management-functions)
10. [Utility Functions](#10-utility-functions)
11. [Baseline & Statistics Functions](#11-baseline--statistics-functions)
12. [Execution Functions](#12-execution-functions)
13. [Global Constants](#13-global-constants)

---

## 1. Classes

### `Reservoir`

Vitter's reservoir sampler for streaming statistics.

**Defined in:** Cell 03 (baseline), Cell 16.1 (online)

```python
class Reservoir:
    def __init__(self, k: int, seed: int)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | `int` | Maximum reservoir capacity |
| `seed` | `int` | RNG seed for `numpy.random.default_rng` |

**Attributes:**
- `k` — capacity
- `n` — total observations seen
- `buf` — `np.ndarray` of dtype `float64`, length `k`
- `rng` — `numpy.random.Generator`

**Methods:**

#### `add(x: float) → None`

Insert observation `x` into the reservoir. If `n ≤ k`, stores directly. Otherwise, replaces a random element with probability `k/n`.

#### `sample() → np.ndarray`

Returns a copy of stored elements. Shape: `(min(n, k),)`, dtype `float64`. Returns empty array if `n == 0`.

#### `size() → int`

Returns `min(n, k)`. Only present in Cell 16.1 version.

---

### `Phase` (Enum)

Governance phase levels.

**Defined in:** Cell 16.1

```python
class Phase(Enum):
    STABLE   = "STABLE"
    CAUTION  = "CAUTION"
    CRITICAL = "CRITICAL"
    LOCKDOWN = "LOCKDOWN"
```

**Ordering:** `STABLE < CAUTION < CRITICAL < LOCKDOWN` (via `PHASE_ORDER` list).

---

### `Keramnych`

Phase controller and decoding cap manager.

**Defined in:** Cell 16.1

```python
class Keramnych:
    def __init__(self, risk_low: float = 0.30, risk_high: float = 0.60, seed: int = 123)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `risk_low` | `float` | `0.30` | Low risk threshold |
| `risk_high` | `float` | `0.60` | High risk threshold |
| `seed` | `int` | `123` | RNG seed |

**Methods:**

#### `phase_from_virtue(V: float) → Phase`

Maps virtue score to governance phase.

| Virtue Range | Phase |
|-------------|-------|
| `V < 0.40` | LOCKDOWN |
| `V < 0.60` | CRITICAL |
| `V < 0.80` | CAUTION |
| `V ≥ 0.80` | STABLE |

**Returns:** `Phase`

#### `decoding_caps(phase: Phase) → dict`

Returns generation parameters for the given phase.

| Phase | max_new_tokens | temperature | top_p |
|-------|---------------|-------------|-------|
| STABLE | 120 | 0.9 | 0.95 |
| CAUTION | 80 | 0.7 | 0.92 |
| CRITICAL | 40 | 0.4 | 0.90 |
| LOCKDOWN | 20 | 0.2 | 0.85 |

**Returns:** `dict` with keys `max_new_tokens`, `temperature`, `top_p`.

---

### `BaselineUpdater`

EMA-based online baseline updater.

**Defined in:** Cell 16.1

#### `update(baseline_state, new_stats, phase) → dict`

Updates baseline statistics with online measurements. Only applies when `phase == Phase.STABLE`.

EMA blend: `0.9 × old + 0.1 × new` for median, MAD, and each quantile.

**Returns:** Updated `baseline_state` dict.

---

### `CCv2Stopping` (StoppingCriteria)

Online generation stopper implementing CC v2.1.3 structural checks.

**Defined in:** Cell 16.2 (inside `run_generation`)

```python
class CCv2Stopping(StoppingCriteria):
    CONSEC_REQUIRED = 3

    def __init__(self, ctx_ref: dict)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx_ref` | `dict` | Reference to live governance context |

**Attributes:**
- `online_kq_history` — list of per-token KQ proxy values
- `kq_regime_history` — copied from `ctx_ref["kq_regime_history"]`
- `stopped` — `bool`, whether generation was halted
- `stop_step` — `int`, step at which generation stopped (-1 if not stopped)
- `stop_reason` — `str`, human-readable reason
- `consec_violations` — `int`, consecutive structural violations
- `noise_floor` — `float`, from `CC_STRUCT_NOISE_FLOOR`
- `js_i_history` — list of per-token JS_intra values

#### `__call__(input_ids, scores, **kwargs) → bool`

Called at each generation step by HuggingFace `generate()`.

**Process:**
1. Computes token-level H, M, JS_intra from latest logits.
2. Computes **simplified KQ proxy**: `KQ_proxy = clamp01((M/M_q95) × (1 - H/H_q95))`. This omits the `(1 - JS_i/JS_i_q95)` factor from the full KQ formula to avoid doubling per-token forward passes (cold/hot JS_intra would require two extra softmax calls at every decoding step). The proxy retains the two dominant signals (entropy and margin) while JS_intra is tracked separately in `js_i_history` for the half-split collapse check.
3. After `≥ MIN_INTRA_TOKENS` steps: computes half-split KQ and JS_i.
4. Runs `compute_delta_KQ_twolayer` with relative JS collapse.
5. Runs `cc_v2_eval` for structural check.
6. If `struct_value > noise_floor`: increments `consec_violations`.
7. If `consec_violations ≥ 3`: sets `stopped = True`, returns `True`.

**Returns:** `True` to stop generation, `False` to continue.

---

## 2. Sensor Functions

### `score_mode_probes(model, tokenizer, prompt: str) → dict`

**Defined in:** Cell 16.1

Computes energy scores for 4 mode probes via cross-entropy loss.

| Probe Key | Probe Text |
|-----------|-----------|
| `comply` | `" Sure. Here are the steps:"` |
| `refuse` | `" I'm sorry, I can't help with that."` |
| `neutral` | `" This topic involves risks and should be handled carefully."` |
| `evasion` | `" This is a complex topic with many considerations."` |

Energy: `E(k) = -loss` where loss is cross-entropy with prompt tokens masked.

Probe tokens truncated to `PROBE_MAX_TOKENS = 64`.

**Returns:** `dict` with keys `{comply, refuse, neutral, evasion}` → `float` energy values.

---

### `compute_intent_hazard(energies: dict) → float`

**Defined in:** Cell 16.2

```python
raw = max(E_comply, E_evasion) - E_refuse
return sigmoid(raw / EI_TAU)
```

**Returns:** `float` in `[0, 1]`. Higher = more hazardous intent.

---

### `detect_response_mode(scores: torch.Tensor) → dict`

**Defined in:** Cell 16.2

Analyzes generation scores to detect refusal/compliance patterns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `scores` | `torch.Tensor` | Stacked generation logits, shape `[T, 1, V]` |

**Returns:** `dict` with:
- `refusal_pressure` — `float` in `[0, 1]`, entropy drop from first to last 10 tokens
- `compliance_pressure` — `float` in `[0, 1]`, margin rise from first to last 10 tokens

---

### `compute_complexity(h_mean: float, m_mean: float, bands: dict) → float`

**Defined in:** Cell 16.2

Lightweight **complexity proxy** (not a formal complexity measure):

```python
complexity_proxy = sigmoid(safe_norm(h, h_med, h_mad)) × sigmoid(-safe_norm(m, m_med, m_mad))
```

Used for monitoring. Does not directly gate CC decisions.

**Returns:** `float` in `(0, 1)`.

---

### `compute_prompt_response_tension(js_ctx: float, js_intra: float) → float`

**Defined in:** Cell 16.2

```python
return clamp01(js_ctx - js_intra)
```

**Returns:** `float` in `[0, 1]`.

---

## 3. JS Divergence Functions

### `js_divergence(p, q) → float`

**Defined in:** Cell 03, Cell 16.2

Top-K restricted Jensen-Shannon divergence.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | `torch.Tensor` | Probability vector |
| `q` | `torch.Tensor` | Probability vector |

Uses `TOP_K_SUPPORT = 200`. Returns `0.0` on NaN/Inf.

**Returns:** `float ≥ 0`.

---

### `stable_js_divergence(p, q, eps=1e-12) → torch.Tensor`

**Defined in:** Cell 03

Float64 numerically stable JS divergence with epsilon smoothing and renormalization.

**Returns:** `torch.Tensor` — JS per batch element (natural log base).

---

### `js_next_token_on_union_support(logits_a, logits_b, topk=200, eps=1e-12) → torch.Tensor`

**Defined in:** Cell 03

JS divergence on union of top-K supports from two logit vectors.

1. Computes top-K indices from each logit vector.
2. Takes union of indices.
3. Extracts softmax probabilities on union support.
4. Calls `stable_js_divergence`.

**Returns:** Scalar `torch.Tensor`.

---

### `js_context_v2_forward` — Two Independent Implementations

> **⚠ Note:** Two independent implementations exist with the same function name. They share the same conceptual purpose (measuring JS context sensitivity) but differ in interface, computation, and return types. They are **not interchangeable**.

#### Baseline Version (Cell 03)

```python
js_context_v2_forward(
    model, tokenizer,
    prompt_text: str,
    perturb_text: str,
    probe_text: str = " Therefore,",
    topk_support: int = 200,
    eps: float = 1e-12,
    device = None
) → (float, dict)
```

**Single forward pass.** Computes one pair of logits (base vs perturbed) at the final token position. Uses `js_next_token_on_union_support` (float64, union top-K). Returns JS value + diagnostics (`max_abs_dp`, `l1_dp`, token lengths).

Used during B0 baseline collection only.

**Returns:** `(js_value: float, diagnostics: dict)`

#### Runtime Version (Cell 16.2)

```python
js_context_v2_forward(
    model, tokenizer,
    prompt: str,
    pert: str,
    positions: int = JS_CTX_POSITIONS,
    lam: float = JS_CTX_LAMBDA,
    mode: str = JS_CTX_V2_MODE,
    probe_text: str = JS_CTX_PROBE_TEXT
) → (float, float, dict)
```

**Multi-position with exponential weighting.** Iterates over `T` probe positions, applies exp-weighted aggregation (top2 or max mode). Uses `js_divergence` (top-K restricted, float32). Returns aggregated value twice (legacy interface) + per-position details.

Used during live generation (Cell 16.2).

**Returns:** `(aggregated_value: float, aggregated_value: float, details: dict)`

---

## 4. Coherence Quality (KQ) Functions

### `compute_KQ(metrics: dict, bands: dict) → float`

**Defined in:** Cell 16.2

```python
KQ = clamp01((M / M_q95) × (1 - H / H_q95) × (1 - JS_i / JS_i_q95))
```

The `clamp01` is a **saturation bound** enforcing KQ's valid `[0, 1]` range — not a numeric correction.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | `dict` | Must contain `h_logit`, `margin`, optionally `js_intra` |
| `bands` | `dict` | Band definitions with quantiles |

**Returns:** `float` in `[0, 1]`.

---

### `compute_KQ_from_arrays(H_arr, M_arr, JS_arr, bands) → float`

**Defined in:** Cell 16.2

Same formula as `compute_KQ` but operates on lists (takes means). Returns `0.0` if arrays empty.

---

### `compute_KQ_stability(metrics: dict, bands: dict) → float`

**Defined in:** Cell 16.2

```python
return clamp01(compute_KQ(metrics, bands))
```

---

### `compute_KQ_hazard(metrics: dict) → float`

**Defined in:** Cell 16.2

```python
return clamp01(0.5 × tension + 0.3 × compliance_pressure + 0.2 × refusal_pressure)
```

**Usage:** Used **exclusively** in the debt update term:
```python
debt += KQ_hazard
```
Does not directly influence phase decisions or CC admissibility.

---

### `_baseline_KQ_from_arrays(H_arr, M_arr, JS_arr, h_q95, m_q95, js_q95) → float`

**Defined in:** Cell 03

Used during baseline collection. Same KQ formula but with explicit q95 parameters instead of bands lookup.

---

## 5. CC Gate Functions

### `cc_v2_eval(delta_KQ, delta_H, delta_Pi, lambda_H, lambda_P, KQ_now, debt_now, struct_noise_floor=None) → dict`

**Defined in:** Cell 16.1

Core Coupling Constraint evaluation — two-component gate.

| Parameter | Type | Description |
|-----------|------|-------------|
| `delta_KQ` | `float` | Structural KQ degradation (from two-layer) |
| `delta_H` | `float` | Entropy increase over baseline MAD |
| `delta_Pi` | `float` | Harm impulse (level-blend from `compute_harm_impulse`) |
| `lambda_H` | `float` | Entropy sensitivity multiplier |
| `lambda_P` | `float` | Harm sensitivity multiplier |
| `KQ_now` | `float` | Current Coherence Quality value |
| `debt_now` | `float` | Current debt value |
| `struct_noise_floor` | `float` or `None` | Override for `CC_STRUCT_NOISE_FLOOR` |

**Internal computation:**

Structural: `struct_value = delta_KQ + lambda_H × delta_H`

Harm: `harm_sat = tanh(lambda_P × delta_Pi)` when `CC_V2_HARM_SATURATION == "tanh"` (default), otherwise `clamp01(lambda_P × delta_Pi)`. The tanh saturation prevents unbounded escalation when lambda_P grows large under high debt.

`harm_value = harm_sat - harm_buffer` where `harm_buffer = CC_HARM_INTENT_BUFFER` (level-scale q95 from B0).

**Returns:** `dict` with keys:
- `enabled` — `True`
- `struct_ok` — `bool`, `struct_value ≤ noise_floor`
- `harm_ok` — `bool`, `harm_value ≤ 0`
- `struct_value` — `float`
- `harm_value` — `float`
- `harm_buffer` — `float`
- `harm_sat` — `float`, the saturated harm signal before buffer subtraction
- `delta_Pi` — `float`
- `lambda_H` — `float`
- `lambda_P` — `float`
- `struct_noise_floor` — `float`

---

### `cc_v2_escalation_phase(struct_value: float, harm_value: float) → Phase`

**Defined in:** Cell 16.1

Maps structural and harm values to escalation phase.

Structural: `>0.03` CAUTION, `>0.10` CRITICAL, `>0.25` LOCKDOWN.
Harm: `>0.02` CAUTION, `>0.06` CRITICAL, `>0.12` LOCKDOWN.

**Returns:** `Phase` — max of structural and harm phases.

---

## 6. Lambda Functions

### `adaptive_lambda_P(debt, intent_hazard, lambda_min=0.1, base=0.3, alpha=4.0) → float`

**Defined in:** Cell 16.1

```python
nu_debt = clamp01((sigmoid(debt) - 0.5) × 2)
nu_P    = clamp01(0.6 × nu_debt + 0.4 × intent_hazard)
return lambda_min + base × exp(alpha × nu_P)
```

**Returns:** `float > 0`.

---

### `adaptive_lambda_H(h_logit, js_intra, bands, eta=1.0) → float`

**Defined in:** Cell 16.1

```python
nu_H  = clamp01((h_logit - h_med) / h_range)
nu_JS = clamp01(js_intra / (js_max + ε))
nu    = clamp01(0.6 × nu_H + 0.4 × nu_JS)
return eta × (1 + nu)
```

**Returns:** `float` in `[η, 2η]`.

---

## 7. Harm Channel Functions

### `compute_harm_impulse(intent_hazard_now, intent_hazard_prev, level_weight=0.7, delta_weight=0.3) → float`

**Defined in:** Cell 16.1 (FIX F5a)

Level-blend harm impulse.

```python
level   = intent_hazard_now
delta   = max(0, intent_hazard_now - intent_hazard_prev)
impulse = level_weight × level + delta_weight × delta
return max(CC_HARM_DELTA_FLOOR, impulse)
```

**Returns:** `float ≥ 0`.

---

### `compute_harm_delta(intent_hazard_now, intent_hazard_prev) → float`

**Defined in:** Cell 16.1

Legacy wrapper — redirects to `compute_harm_impulse`.

---

### `_cc_v2_harm_buffer(KQ_now: float, debt_now: float) → float`

**Defined in:** Cell 16.1 (FIX F5b)

Returns `CC_HARM_INTENT_BUFFER` (q95 of intent_hazard from B0). Parameters `KQ_now` and `debt_now` are accepted for interface compatibility but not used in v2.1.3.

---

### `_cc_v2_saturate_harm(x: float) → float`

**Defined in:** Cell 16.1

Applies saturation function to harm signal. If `CC_V2_HARM_SATURATION == "tanh"` (default): returns `tanh(max(0, x))`. Otherwise: `clamp01(x)`.

---

### `_compute_harm_intent_baseline(baseline_data: dict, percentile: float = 95.0) → float`

**Defined in:** Cell 16.1

Computes harm buffer from B0 `intent_samples`. Falls back to `CC_INTENT_HAZARD_BASELINE` if samples unavailable.

**Returns:** `float ≥ 1e-4`.

---

## 8. Structural Channel Functions

### `compute_delta_KQ_twolayer(KQ_intra_first, KQ_intra_second, KQ_current, kq_regime_history, JS_i_first=None, JS_i_second=None, w_intra=0.6, w_regime=0.4) → dict`

**Defined in:** Cell 16.1

Two-layer structural delta with relative JS collapse gating (FIX F6).

| Parameter | Type | Description |
|-----------|------|-------------|
| `KQ_intra_first` | `float` | KQ of first half of generation |
| `KQ_intra_second` | `float` | KQ of second half of generation |
| `KQ_current` | `float` | Overall KQ of current generation |
| `kq_regime_history` | `list` | History of KQ values across generations |
| `JS_i_first` | `float` or `None` | Mean JS_intra of first half |
| `JS_i_second` | `float` or `None` | Mean JS_intra of second half |
| `w_intra` | `float` | Weight for intra layer (default 0.6) |
| `w_regime` | `float` | Weight for regime layer (default 0.4) |

**Returns:** `dict` with keys:
- `delta_KQ` — final combined delta
- `delta_intra` — gated intra delta (0 if suppressed)
- `delta_intra_raw` — ungated intra delta
- `delta_regime` — regime drift delta
- `weighted_intra` — `w_intra × delta_intra`
- `weighted_regime` — `w_regime × delta_regime`
- `dominant` — `"intra"` or `"regime"`
- `kq_regime_mean` — mean of regime window
- `kq_regime_n` — length of regime history
- `kq_regime_cold` — `True` if insufficient samples
- `KQ_intra_first`, `KQ_intra_second`, `KQ_current` — inputs echoed
- `w_intra`, `w_regime` — weights echoed
- `is_js_collapse` — `bool`, whether JS collapse detected
- `JS_i_first`, `JS_i_second` — inputs echoed
- `js_collapse_method` — `"relative"`
- `js_collapse_rel_eps` — threshold used
- `js_i_min_floor` — floor used
- `delta_intra_suppressed` — `bool`, `True` if intra was zeroed

---

## 9. State Management Functions

### `make_ctx() → dict`

**Defined in:** Cell 16.1

Creates fresh governance context.

**Returns:** `dict` with keys:
- `phase` — `Phase.STABLE`
- `debt` — `0.0`
- `virtue` — `1.0`
- `js_ctx_ema` — `0.0`
- `prev_KQ` — `None`
- `prev_H` — `None`
- `prev_harm` — `float(CC_INTENT_HAZARD_BASELINE)`
- `cum_Pi` — `0.0`
- `cc_violations` — `0`
- `cc_value_last` — `None`
- `lambda_P_last` — `None`
- `lambda_H_last` — `None`
- `kq_history` — `[]`
- `kq_regime_history` — `[]`

---

### `reset_ctx(ctx: dict, hard_reset: bool = False) → None`

**Defined in:** Cell 16.1

If `hard_reset`: replaces entire context with `make_ctx()`.
Otherwise: resets only `phase`, `debt`, `virtue` to defaults.

---

### `_copy_ctx(ctx: dict) → dict`

**Defined in:** Cell 16.1

Shallow copy of context. Lists are copied; other values are shared.

---

### `is_first_step(ctx: dict) → bool`

**Defined in:** Cell 16.1

Returns `True` if `ctx["prev_KQ"] is None`.

---

## 10. Utility Functions

### `safe_norm(x: float, med: float, mad: float) → float`

**Defined in:** Cell 16.1

```python
return (x - med) / (mad + ε)
```

Returns `0.0` if `mad ≤ 0`.

---

### `clamp01(x: float) → float`

**Defined in:** Cell 16.1

```python
return max(0.0, min(1.0, x))
```

A **saturation bound** enforcing `[0, 1]` range for metrics that are defined on this domain.

---

### `sigmoid(z: float) → float`

**Defined in:** Cell 16.1

```python
return 1.0 / (1.0 + exp(-z))
```

---

### `get_quantile(bands: dict, metric: str, q: float) → float`

**Defined in:** Cell 16.1

Retrieves quantile value from bands dict. Key format: `q{int(q*100):02d}`. Returns `EPS` on failure.

---

### `baseline_median(metric: str) → float`

**Defined in:** Cell 16.1

Returns `baseline[metric]["median"]`. Returns `0.0` on failure.

---

### `baseline_q95(metric: str, default: float = 1.0) → float`

**Defined in:** Cell 16.1

Returns `baseline[metric]["quantiles"]["q95"]`. Returns `default` on failure.

---

### `to_device(tensor_dict: dict) → dict`

**Defined in:** Cell 16.1

Moves all tensors in dict to model's device.

---

### `phase_max(p1: Phase, p2: Phase) → Phase`

**Defined in:** Cell 16.1

Returns the more severe of two phases based on `PHASE_ORDER`.

---

### `set_all_seeds(seed: int) → None`

**Defined in:** Cell 16.1

Sets seeds for `random`, `numpy`, `torch`, and CUDA (if available). Enables `cudnn.deterministic`, disables `cudnn.benchmark`.

---

### `compute_hash(items: list) → str`

**Defined in:** Cell 04

SHA-256 hash of sorted string representations of list items.

---

## 11. Baseline & Statistics Functions

### `robust_median_mad(arr) → dict`

**Defined in:** Cell 03

Returns `{"median": float, "mad": float, "samples": int}`. MAD scaled by 1.4826.

---

### `quantiles(arr, qs=(0.05, 0.20, 0.50, 0.80, 0.95)) → dict`

**Defined in:** Cell 03

Returns dict of quantile values. Keys: `q05, q20, q50, q80, q95`. Filters non-finite values.

---

### `recompute_stats_from_reservoir(res: Reservoir) → dict`

**Defined in:** Cell 16.1

Computes full statistics from reservoir sample. Returns dict with `median`, `mad`, `samples`, `quantiles`.

---

### `estimate_intent_hazard_baseline_percentile(b0: dict, percentile: float = 95.0) → float`

**Defined in:** Cell 04

Computes CC_INTENT_HAZARD_BASELINE from `margin_logit_samples`:

```python
norm_margin = margins / (margins + 1 + ε)
hazards = 1 - norm_margin
return percentile(hazards, 95)
```

Requires `≥ 10` samples.

---

### `estimate_delta_pi_baseline_percentile(b0: dict, percentile: float = 99.0) → float`

**Defined in:** Cell 04

Returns 99th percentile of `delta_pi_samples` from baseline. Requires `≥ 10` samples.

---

### `estimate_kq_baseline(b0: dict) → float`

**Defined in:** Cell 04

Computes `CC_KQ_BASELINE` from B0 medians and q95 values.

---

### `estimate_kq_noise_floor(b0: dict) → float`

**Defined in:** Cell 04

Legacy per-token KQ noise floor: `MAD(margin)/M_q95 + MAD(h_logit)/H_q95`. Minimum `0.01`.

---

### `estimate_struct_noise_floor(b0: dict, percentile: float = 95.0) → float`

**Defined in:** Cell 04

CC v2.1 struct noise floor: q95 of `struct_value_samples`. Minimum `1e-6`. Requires `≥ 10` samples.

---

### `build_bands_from_quantiles(baseline: dict) → dict`

**Defined in:** Cell 04

Constructs empirical band definitions for all metrics.

**Returns:** `dict` with keys `h_logit`, `margin`, `js_context`, `js_intra`, `complexity`. Each contains `B_STABLE`, `B_TAIL`, `B_EXTREME` boundaries (where applicable), `baseline` statistics, and mode description.

Special cases:
- `margin` — inverted (lower = higher risk). Includes `M_min`.
- `js_intra` — frozen detector mode with `frozen_threshold = max(q95, median + 3×MAD)`. Includes `JS_max`.
- `complexity` — online only, no pre-computed bands.

---

## 12. Execution Functions

### `run_generation(prompt: str, ctx: dict, is_base: bool) → tuple`

**Defined in:** Cell 16.2

Main execution function. Runs one prompt through the model with full CC instrumentation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | Input prompt text |
| `ctx` | `dict` | Governance context (modified in place) |
| `is_base` | `bool` | `True` for BASE (measure only), `False` for WRAPPER |

**Process:**
1. Selects decoding caps: BASE uses fixed (120/0.9/0.95), WRAPPER uses `keramnych.decoding_caps(phase)`.
2. Attaches `CCv2Stopping` criteria (WRAPPER only).
3. Runs `model.generate()` with `output_scores=True`.
4. Computes per-token: H, M (probability-gap under generation temperature), JS_intra.
5. Computes half-split KQ (if `≥ MIN_INTRA_TOKENS`).
6. Computes JS context (5 perturbations, aggregated).
7. Computes complexity proxy, response mode, intent hazard.
8. Runs two-layer structural delta.
9. Updates debt, virtue, JS_ctx_ema.
10. Evaluates CC gate (`cc_v2_eval`).
11. Determines phase (intent override → JS context → virtue → CC escalation).
12. Updates context state.
13. Conditionally updates online baseline (if STABLE + high virtue + low debt).

**State mutation invariants:**

```
BASE mode (BASE_MEASURE_ONLY = True) guarantees:
  ✓ Updates: prev_KQ, prev_H, prev_harm, cum_Pi, cc_violations, kq_history
  ✗ No mutation of: debt
  ✗ No mutation of: virtue
  ✗ No mutation of: phase
  ✗ No mutation of: js_ctx_ema
  ✗ No mutation of: kq_regime_history

WRAPPER mode mutates ALL context fields:
  ✓ debt, virtue, phase, js_ctx_ema
  ✓ prev_KQ, prev_H, prev_harm, cum_Pi, cc_violations
  ✓ kq_history, kq_regime_history (STABLE phase only if REGIME_KQ_STABLE_ONLY)
```

**Returns:** `tuple` of:
- `text` — `str`, generated text
- `token_count` — `int`, number of generated tokens
- `duration` — `float`, wall time in seconds
- `metrics` — `dict`, full metrics including `cc_v2` sub-dict
- `phase` — `str`, final phase value

---

### `_score_mode_probes_baseline(model, tokenizer, prompt: str) → dict`

**Defined in:** Cell 03

Baseline version of mode probe scoring (identical logic to `score_mode_probes`). Used during B0 collection.

---

### `_compute_intent_hazard_from_energies(energies: dict) → float`

**Defined in:** Cell 03

Baseline version of intent hazard computation. Same formula as `compute_intent_hazard` but uses `np.exp` directly.

---

## 13. Global Constants

### Structural Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CC_V2_ENABLED` | `True` | CC v2 gate active |
| `CC_V2_STRUCT_W_INTRA` | `0.6` | Intra-generation weight in two-layer delta |
| `CC_V2_STRUCT_W_REGIME` | `0.4` | Regime drift weight in two-layer delta |
| `CC_V2_STRUCT_CAUTION_THR` | `0.03` | Structural CAUTION escalation threshold |
| `CC_V2_STRUCT_CRIT_THR` | `0.10` | Structural CRITICAL escalation threshold |
| `CC_V2_STRUCT_LOCK_THR` | `0.25` | Structural LOCKDOWN escalation threshold |
| `MIN_INTRA_TOKENS` | `12` | Minimum tokens for half-split analysis |
| `REGIME_KQ_WINDOW` | `5` | Regime KQ history window size |
| `REGIME_KQ_MIN_SAMPLES` | `2` | Minimum samples for regime drift |
| `REGIME_KQ_STABLE_ONLY` | `True` | Only update regime history in STABLE phase |

### Harm Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CC_V2_HARM_SATURATION` | `"tanh"` | Harm saturation function |
| `CC_HARM_LEVEL_WEIGHT` | `0.7` | Level component weight in harm impulse |
| `CC_HARM_DELTA_WEIGHT` | `0.3` | Delta component weight in harm impulse |
| `CC_HARM_DELTA_FLOOR` | `0.0` | Minimum harm impulse value |
| `CC_V2_HARM_CAUTION_THR` | `0.02` | Harm CAUTION escalation threshold |
| `CC_V2_HARM_CRIT_THR` | `0.06` | Harm CRITICAL escalation threshold |
| `CC_V2_HARM_LOCK_THR` | `0.12` | Harm LOCKDOWN escalation threshold |
| `CC_LAMBDA_P_BASE` | `0.3` | Lambda_P base value |
| `CC_LAMBDA_P_ALPHA` | `4.0` | Lambda_P exponential scaling factor |
| `CC_LAMBDA_P_MIN` | `0.1` | Lambda_P minimum value |
| `CC_NU_P_DEBT_WEIGHT` | `0.6` | Debt contribution to nu_P |
| `CC_NU_P_HAZARD_WEIGHT` | `0.4` | Hazard contribution to nu_P |
| `W_EI` | `2.5` | Intent hazard weight in debt accumulation |
| `EI_TAU` | `0.35` | Intent hazard sigmoid temperature |

### JS Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `JS_CTX_POSITIONS` | `12` | Number of probe positions |
| `JS_CTX_LAMBDA` | `0.15` | Exponential weight decay |
| `JS_CTX_V2_MODE` | `"top2"` | Position aggregation mode |
| `JS_CTX_PROBE_TEXT` | `" Therefore,"` | Probe continuation text |
| `JS_CTX_PERT_AGG` | `"max"` | Multi-perturbation aggregation |
| `JS_CTX_PRIMARY_SENSOR` | `True` | JS context as primary phase sensor |
| `JS_CTX_PHASE_Q_CAUTION` | `0.90` | Caution quantile for JS phase |
| `JS_CTX_PHASE_Q_LOCKDOWN` | `0.97` | Lockdown quantile for JS phase |
| `JS_CTX_NOISE_FLOOR` | `1e-12` | JS context noise floor |
| `JS_TAU` | `0.15` | JS context EMA smoothing rate |
| `JS_COLLAPSE_REL_EPS` | `0.25` | Relative JS collapse threshold (25%) |
| `JS_I_MIN_FLOOR` | `0.020` | Minimum JS_intra for collapse signal |
| `JS_COLLAPSE_EPS` | `0.003` | Legacy absolute JS collapse epsilon |
| `TOP_K_SUPPORT` | `200` | Top-K support for JS computation |
| `JS_INTRA_FROZEN_THRESHOLD` | (from bands) | `max(q95, median + 3×MAD)` of JS_intra |

### Governance Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CC_LAMBDA_H_ETA` | `1.0` | Lambda_H base scaling factor |
| `R_LOW` | `0.30` | Keramnych low risk threshold |
| `R_HIGH` | `0.60` | Keramnych high risk threshold |
| `DEBT_LEAK` | `0.02` | Debt decay rate per step |
| `DEBT_EPS` | `1e-6` | Debt epsilon threshold |
| `MAINTENANCE_DEBT_FLOOR` | `0.0` | Minimum debt value |
| `VIRTUE_ALPHA` | `0.85` | Virtue EMA smoothing factor |
| `VIRTUE_DISCHARGE_BETA` | `0.08` | Virtue debt discharge rate |
| `VIOLATION_DEMPFER_DECREMENT` | `1` | Violation counter decrement on admissible step |
| `EPS` | `1e-12` | Global epsilon for numerical stability |

### Runtime Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_NEW_TOKENS` | `100` | Default max tokens (Cell 16.1 global) |
| `PROBE_MAX_TOKENS` | `64` | Max tokens for mode probes |
| `BASE_MEASURE_ONLY` | `True` | BASE mode: observe only, no state mutation |
| `BASE_REPORT_WOULD_BE_STATE` | `True` | Report hypothetical state in BASE metrics |
| `ENABLE_ONLINE_BASELINE_UPDATE` | `False` | Enable online baseline EMA update |

### Baseline & Reservoir Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `BASELINE_SEED` | `424242` | Seed for baseline collection |
| `RES_K` | `2048` | Baseline reservoir capacity |
| `ONLINE_RES_K` | `512` | Online reservoir capacity |
| `ONLINE_SEED` | `123456` | Online reservoir seed |

### Empirical Baselines (computed at runtime from B0 data)

| Variable | Source | Description |
|----------|--------|-------------|
| `CC_INTENT_HAZARD_BASELINE` | Cell 04 | q95 of hazard proxy from margins |
| `CC_DELTA_PI_BASELINE` | Cell 04 | q99 of delta_pi (legacy, kept for reference) |
| `CC_KQ_BASELINE` | Cell 04 | KQ from B0 medians |
| `CC_KQ_NOISE_FLOOR` | Cell 04 | Legacy per-token noise |
| `CC_STRUCT_NOISE_FLOOR` | Cell 04 | q95 of struct_value on B0 (primary noise floor) |
| `CC_HARM_INTENT_BUFFER` | Cell 16.1 | q95 of intent_hazard from B0 (level-scale harm buffer) |
