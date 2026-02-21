# Mathematical Foundations — QTStoic CC v2.1.3

All formulas below are extracted directly from the codebase. No external derivations.

**Terminology:**
- **CC** — Coupling Constraint (the two-channel admissibility gate)
- **KQ** — Coherence Quality (the composite metric measuring generation coherence)

---

## 1. Core Sensors

### 1.1 Token-Level Entropy

For probability distribution `p` over vocabulary at generation step `t`, with temperature `T_jitter`:

```
p_gen = softmax(logits / T_jitter)
H(t) = -Σ_i p_gen(i) · log(p_gen(i) + ε)
```

where `ε = 1e-12`.

### 1.2 Margin (Probability-Gap under T_jitter)

Top-2 probability gap computed from the **temperature-scaled** distribution `p_gen`, not from raw logits:

```
p_gen = softmax(logits / T_jitter)
margin(t) = max(0, p_gen[top1] - p_gen[top2])
```

The `max(0, ·)` clamp is a **saturation bound** ensuring the metric stays in its valid domain `[0, 1]` — not a correction for negative values (the difference of two probabilities can't be negative, but floating-point arithmetic can produce negligible negative residuals).

### 1.3 JS Intra-Temperature Divergence

Cold/hot curvature measurement:

```
p_cold = softmax(logits / 0.6)
p_hot  = softmax(logits / 1.4)
JS_intra(t) = JS(p_cold, p_hot)
```

Using top-K restricted JS divergence (K=200):

```
JS(p, q) = 0.5 · [KL(p_k || m_k) + KL(q_k || m_k)]
```

where:
- `m_k = 0.5 · (p_k + q_k)`
- `p_k, q_k` are renormalized over the top-K indices of `m`
- Returns `0.0` if result is NaN or Inf

### 1.4 Stable JS Divergence (float64)

Used in `js_next_token_on_union_support`:

```
p, q ← clamp(·, min=ε), then renormalize
m = 0.5 · (p + q)
JS = 0.5 · [Σ p · log(p/m) + Σ q · log(q/m)]
```

Computed in float64 with epsilon-smoothing. Support restricted to union of top-K indices from both distributions.

### 1.5 Repetition Rate

```
rep(t) = 1.0  if token(t) == token(t-1)
         0.0  otherwise
```

---

## 2. JS Context Sensitivity (v2 Forward Probe)

### 2.1 Single Perturbation

Given prompt `P`, perturbation text `pert`, and probe text (default: `" Therefore,"`):

```
a_text = P + probe_prefix(t)
b_text = P + pert + probe_prefix(t)

logits_a = model(a_text).logits[0, -1, :]
logits_b = model(b_text).logits[0, -1, :]

JS_ctx(t) = JS_union_topk(logits_a, logits_b, K=200)
```

Iterated over `T` positions (T = min(JS_CTX_POSITIONS=12, len(probe_ids)+1)).

### 2.2 Exponential Weighting

```
w(t) = exp(-λ · t) / Σ_j exp(-λ · j)
```

where `λ = JS_CTX_LAMBDA = 0.15`.

### 2.3 Aggregation Modes

**top2 mode** (default `JS_CTX_V2_MODE`):
```
i1 = argmax(d)
i2 = argmax₂(d)
JS_ctx = (w[i1]·d[i1] + w[i2]·d[i2]) / (w[i1] + w[i2] + ε)
```

**max mode**:
```
JS_ctx = max(w · d)
```

### 2.4 Multi-Perturbation Aggregation

5 perturbations from `JS_CTX_PERT_SET`. Aggregation by `JS_CTX_PERT_AGG`:

- `"top2"`: average of 2 highest values across perturbations
- `"max"`: maximum value

Result floored to 0 if `|JS_ctx_raw| < JS_CTX_NOISE_FLOOR` (1e-12).

---

## 3. Coherence Quality (KQ)

### 3.1 Point KQ

```
KQ = (M / M_q95) · (1 - H / H_q95) · (1 - JS_i / JS_i_q95)
```

Clamped to `[0, 1]` via `clamp01` — a saturation bound enforcing the metric's valid range, not a numeric patch. All `q95` values from empirical B0 baseline.

### 3.2 KQ from Arrays (Half-Split)

For token arrays `H[], M[], JS_i[]`:

```
KQ(arr) = (mean(M) / M_q95) · (1 - mean(H) / H_q95) · (1 - mean(JS_i) / JS_i_q95)
```

### 3.3 KQ Baseline

From B0 medians:

```
KQ_baseline = (M_med / M_q95) · (1 - H_med / H_q95) · (1 - JS_med / JS_q95)
```

### 3.4 KQ Stability

```
KQ_stability = clamp01(KQ)
```

### 3.5 KQ Hazard

```
KQ_hazard = 0.5 · tension + 0.3 · compliance_pressure + 0.2 · refusal_pressure
```

Used exclusively in the debt update term (Section 12). Does not directly influence phase or CC admissibility.

---

## 4. Intent Hazard (Mode Probes)

### 4.1 Energy Scoring

For each mode probe `k ∈ {comply, refuse, neutral, evasion}`:

```
E(k) = -CrossEntropyLoss(model(prompt + probe_k), labels=probe_k_tokens)
```

Labels mask prompt tokens with `-100`. Probe tokens limited to `PROBE_MAX_TOKENS = 64`.

### 4.2 Intent Hazard

```
raw = max(E_comply, E_evasion) - E_refuse
intent_hazard = σ(raw / EI_TAU)
```

where `EI_TAU = 0.35` and `σ` is the standard sigmoid.

### 4.3 Baseline Intent Hazard (Cell 04)

Margin-based proxy for baseline:

```
norm_margin = margin / (margin + 1 + ε)
hazard = 1 - norm_margin
CC_INTENT_HAZARD_BASELINE = percentile(hazard, 95)
```

---

## 5. Complexity Proxy

A lightweight **proxy** for prompt difficulty — not a formal complexity measure:

```
H_norm = (H - H_med) / (H_mad + ε)
M_norm = (M - M_med) / (M_mad + ε)
complexity_proxy = σ(H_norm) · σ(-M_norm)
```

High entropy + low margin → higher proxy value. Used for monitoring; does not directly gate CC decisions.

---

## 6. Prompt-Response Tension

```
tension = clamp01(JS_context - JS_intra)
```

---

## 7. Response Mode Detection

Over generation scores (per-token softmax):

```
ent_drop = mean(H[:10]) - mean(H[-10:])
mar_rise = mean(M[-10:]) - mean(M[:10])

refusal_pressure    = clamp01(ent_drop / (|mean(H[:10])| + ε))
compliance_pressure = clamp01(mar_rise / (|mean(M[:10])| + ε))
```

---

## 8. Two-Layer Structural Delta (compute_delta_KQ_twolayer)

### 8.1 Layer 1 — Intra-Generation

```
delta_intra_raw = max(0, KQ_first_half - KQ_second_half)
```

Gated by **relative JS collapse** (FIX F6):

```
if JS_i_first < JS_I_MIN_FLOOR (0.020):
    is_js_collapse = False    // below noise
else:
    is_js_collapse = (JS_i_second > JS_i_first × (1 + JS_COLLAPSE_REL_EPS))
    // JS_COLLAPSE_REL_EPS = 0.25 (25% relative rise required)

delta_intra = delta_intra_raw  if is_js_collapse
              0.0              otherwise
```

### 8.2 Layer 2 — Regime Drift

```
kq_regime_mean = mean(kq_regime_history[-REGIME_KQ_WINDOW:])
delta_regime = max(0, kq_regime_mean - KQ_current)
```

Requires `≥ REGIME_KQ_MIN_SAMPLES (2)` in history. Only updated during STABLE phase if `REGIME_KQ_STABLE_ONLY = True`.

### 8.3 Combined

```
weighted_intra  = w_intra  × delta_intra     // w_intra  = 0.6
weighted_regime = w_regime × delta_regime     // w_regime = 0.4
delta_KQ = max(weighted_intra, weighted_regime)
dominant = "intra" if weighted_intra ≥ weighted_regime else "regime"
```

---

## 9. Harm Impulse (FIX F5)

### 9.1 Level-Blend (v2.1.3)

```
level   = intent_hazard_now
delta   = max(0, intent_hazard_now - intent_hazard_prev)
impulse = 0.7 × level + 0.3 × delta
impulse = max(CC_HARM_DELTA_FLOOR, impulse)     // floor = 0.0
```

### 9.2 Harm Buffer (v2.1.3)

```
harm_buffer = CC_HARM_INTENT_BUFFER
            = q95(intent_hazard samples from B0)
```

Level-scale (not delta-scale). Replaces legacy `CC_V2_DELTA_PI_Q = q99(delta_pi)`.

---

## 10. Adaptive Lambda Functions

### 10.1 Lambda_P (Harm Sensitivity)

```
nu_debt = clamp01((σ(debt) - 0.5) × 2)
nu_P    = clamp01(0.6 × nu_debt + 0.4 × intent_hazard)
λ_P     = λ_min + base × exp(α × nu_P)
```

Constants: `λ_min = 0.1`, `base = 0.3`, `α = 4.0`.

### 10.2 Lambda_H (Entropy Sensitivity)

```
h_range = max(H_q95 - H_med, ε)
nu_H    = clamp01((H - H_med) / h_range)
nu_JS   = clamp01(JS_intra / (JS_max + ε))
nu      = clamp01(0.6 × nu_H + 0.4 × nu_JS)
λ_H     = η × (1 + nu)
```

where `η = CC_LAMBDA_H_ETA = 1.0`.

---

## 11. CC Gate Evaluation (cc_v2_eval)

### 11.1 Structural Value

```
struct_value = delta_KQ + λ_H × delta_H
struct_ok    = (struct_value ≤ CC_STRUCT_NOISE_FLOOR)
```

`delta_H`:
```
delta_H = max(0, (H_now - H_prev) - H_mad_baseline)
        = 0  on first step
```

### 11.2 Harm Value

`delta_Pi` is the **level-blend harm impulse** from Section 9.1:
```
delta_Pi = compute_harm_impulse(intent_hazard_now, intent_hazard_prev)
         = 0.7 × intent_hazard_now + 0.3 × max(0, intent_hazard_now - intent_hazard_prev)
```

This impulse is then scaled and saturated:
```
harm_sat   = tanh(λ_P × max(CC_HARM_DELTA_FLOOR, delta_Pi))
```

The `tanh` saturation bounds the scaled impulse to `[0, 1)`, preventing unbounded escalation when `λ_P` grows large under high debt. The saturation mode is controlled by `CC_V2_HARM_SATURATION` (default: `"tanh"`; alternative: `clamp01`).

```
harm_value = harm_sat - harm_buffer
harm_ok    = (harm_value ≤ 0)
```

`harm_buffer = CC_HARM_INTENT_BUFFER` — the B0 noise floor for intent hazard (Section 9.2).

### 11.3 Admissibility

```
cc_admissible = struct_ok AND harm_ok
```

---

## 12. Debt Dynamics

```
debt(t+1) = (1 - DEBT_LEAK) × debt(t)
           + max(0, KQ_med - KQ_stability) × JS_ctx_ema
           + KQ_hazard                                      ← Section 3.5
           + W_EI × intent_hazard
```

Constants: `DEBT_LEAK = 0.02`, `W_EI = 2.5`, `MAINTENANCE_DEBT_FLOOR = 0.0`.

### 12.1 JS Context EMA

```
JS_ctx_ema(t+1) = (1 - JS_TAU) × JS_ctx_ema(t) + JS_TAU × tension
```

where `JS_TAU = 0.15` and `tension = clamp01(JS_ctx - JS_intra)`.

### 12.2 Virtue

```
V_inst = KQ_stability / (KQ_stability + debt + ε)
V(t+1) = α × V(t) + (1 - α) × V_inst
```

where `α = VIRTUE_ALPHA = 0.85`.

### 12.3 Debt Regulation

```
debt_regulated = max(floor, debt × (1 - β × V))
```

where `β = VIRTUE_DISCHARGE_BETA = 0.08`.

---

## 13. Phase Escalation

### 13.1 CC-Based Escalation

Structural thresholds:
```
CAUTION:  struct_value > 0.03
CRITICAL: struct_value > 0.10
LOCKDOWN: struct_value > 0.25
```

Harm thresholds:
```
CAUTION:  harm_value > 0.02
CRITICAL: harm_value > 0.06
LOCKDOWN: harm_value > 0.12
```

Final CC phase = `max(phase_struct, phase_harm)`.

### 13.2 Virtue-Based Phase (Keramnych)

```
V < 0.40 → LOCKDOWN
V < 0.60 → CRITICAL
V < 0.80 → CAUTION
V ≥ 0.80 → STABLE
```

### 13.3 Overall Phase

```
phase = max(intent_override, js_context_override, virtue_phase, cc_escalation)
```

### 13.4 Violation Counter (FIX F4)

```
if cc_admissible:
    violations = max(0, violations - VIOLATION_DEMPFER_DECREMENT)   // decrement = 1
else:
    violations = violations + 1
```

---

## 14. Decoding Caps (Keramnych)

| Phase | max_new_tokens | temperature | top_p |
|-------|---------------|-------------|-------|
| STABLE | 120 | 0.9 | 0.95 |
| CAUTION | 80 | 0.7 | 0.92 |
| CRITICAL | 40 | 0.4 | 0.90 |
| LOCKDOWN | 20 | 0.2 | 0.85 |

BASE always uses: 120 tokens, T=0.9, top_p=0.95.

---

## 15. Online Stopper (CCv2Stopping)

### 15.1 Simplified KQ Proxy

The stopper uses a **two-factor KQ proxy** instead of the full three-factor KQ:

```
KQ_proxy = clamp01((M / M_q95) × (1 - H / H_q95))
```

**Rationale:** The full KQ formula (Section 3.1) includes a `(1 - JS_i / JS_i_q95)` term requiring cold/hot curvature computation (two extra softmax calls per token). The stopper executes at every decoding step, so this cost would approximately double per-token latency. The proxy retains entropy and margin — the two dominant signals — while JS_intra is tracked separately in `js_i_history` for the half-split collapse check.

### 15.2 Stopping Logic

After `≥ MIN_INTRA_TOKENS (12)` steps:

```
KQ_first  = mean(kq_history[:mid])
KQ_second = mean(kq_history[mid:])
JS_i_first  = mean(js_i_history[:mid])
JS_i_second = mean(js_i_history[mid:])
```

Runs `compute_delta_KQ_twolayer` → `cc_v2_eval`. If `struct_value > noise_floor`:
```
consec_violations += 1
if consec_violations ≥ 3:
    STOP generation
```

Reset to 0 on any admissible step.

---

## 16. Reservoir Sampling

Vitter's Algorithm R:

```
if n ≤ K:
    buf[n-1] = x
else:
    j = random_integer(1, n)
    if j ≤ K:
        buf[j-1] = x
```

Ensures uniform probability `K/n` for each element. Used with `K=2048` (baseline) and `K=512` (online).

---

## 17. Robust Statistics

### Median Absolute Deviation (MAD)

```
MAD = median(|x_i - median(x)|) × 1.4826
```

Scale factor 1.4826 makes MAD consistent with standard deviation for normal distributions.

### Normalization

```
safe_norm(x, med, mad) = (x - med) / (mad + ε)
```

---

## 18. Baseline Update (Online, STABLE-only)

EMA blending when `phase = STABLE, virtue > 0.9, debt ≤ ε`:

```
median_new = 0.9 × median_old + 0.1 × median_online
mad_new    = 0.9 × mad_old    + 0.1 × mad_online
q_new      = 0.9 × q_old      + 0.1 × q_online
```

Controlled by `ENABLE_ONLINE_BASELINE_UPDATE` flag (default: `False`).
