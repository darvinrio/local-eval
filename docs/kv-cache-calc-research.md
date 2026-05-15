# KV-Cache Sizing: Qwen 3.6, Qwen 3.5, Gemma 4, DeepSeek (March 2026+)

**Sources:** [Raschka — KV Cache Calculations](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/) · [Raschka — LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) · [Raschka — Attention Variants](https://magazine.sebastianraschka.com/p/visual-attention-variants)

**Model configs fetched from HuggingFace:** Qwen/Qwen3.6-27B, Qwen/Qwen3.6-35B-A3B, Qwen/Qwen3.5-32B, google/gemma-4-31b-it, google/gemma-4-26b-a4b-it, deepseek-ai/DeepSeek-V3

**Authored by**: Hermes Agent

---

## 1. KV-Cache Formulas by Attention Type

### Standard Full Attention (MHA / GQA)

    KV_bytes = 2 × L × H_kv × D × S × B

- **2** = K and V stored separately
- **L** = num_hidden_layers
- **H_kv** = num_key_value_heads
- **D** = head_dim = hidden_size / num_attention_heads
- **S** = sequence length (context tokens)
- **B** = bytes per value: 2 (BF16/FP16), 1 (INT8), 0.5 (INT4)

### Sliding Window Attention (SWA)

    KV_SWA = 2 × L × H_kv × D × min(S, W) × B

Memory caps at window size W.

### Hybrid Interleaved (Full + SWA) — Gemma 4 Pattern

Gemma 4 has `attention_k_eq_v = True` for its **global/full** layers, meaning K and V share the same tensor (1× not 2×). The global layers also use separate head parameters (`num_global_key_value_heads`, `global_head_dim`).

    KV_hybrid = B × [ Σ_full(1 × H_global_kv × D_global × S) + Σ_SWA(2 × H_kv × D × min(S,W)) ]

At S » W: growth rate = (L_full / L) × full-attention-rate.

### Gated DeltaNet / Linear Attention — Qwen 3.6 Pattern

Qwen 3.6 uses **Gated DeltaNet** layers (linear attention, RNN-like) interleaved with **Gated Attention** layers (standard full attention with RoPE).

**Linear attention layers** do NOT use a KV-cache in the traditional sense. Instead they maintain a **fixed-size state matrix** per layer:

    State = (H_QK_linear × D_QK_linear + H_V_linear × D_V_linear) × B per layer

This is O(1) with respect to sequence length. The state matrix is updated incrementally for each new token.

**Full attention layers** do use a standard KV-cache:

    KV_full = 2 × H_kv × D × S × B per full-attention layer

For Qwen 3.6: 3 DeltaNet layers + 1 Gated Attention layer, repeating.

**Total KV-cache for Qwen 3.6 hybrid:**

    KV_total = L_full_layers × 2 × H_kv × D × S × B + L_linear_layers × (H_QK_linear × D_QK_linear + H_V_linear × D_V_linear) × B

Where:
- L_full_layers = L / full_attention_interval (every 4th layer)
- L_linear_layers = L - L_full_layers
- H_QK_linear = linear_num_key_heads (16)
- D_QK_linear = linear_key_head_dim (128)
- H_V_linear = linear_num_value_heads (48 for 27B, 32 for 35B-A3B)
- D_V_linear = linear_value_head_dim (128)

**Key insight:** The linear attention layers have ZERO growth with context. Only the full-attention layers contribute O(S) KV-cache. This is fundamentally different from sliding window attention.

### Grouped Query Attention (GQA)

    KV_GQA = 2 × L × H_kv × D × S × B    where H_kv < H_q

Savings vs MHA: factor of H_q / H_kv.

### Multi-Query Attention (MQA)

    KV_MQA = 2 × L × 1 × D × S × B

### Multi-Latent Attention (MLA) — DeepSeek-V3

MLA compresses KV via low-rank projections into a single latent vector:

    KV_MLA_per_layer_per_token = (kv_lora_rank + qk_rope_head_dim) × B

Total: KV_MLA = L × S × (kv_lora_rank + qk_rope_head_dim) × B

DeepSeek-V3: kv_lora_rank=512, qk_rope_head_dim=64 → 576 dims/token/layer
vs standard 2×H×D = 2×128×128 = 32,768 → **~57× compression**

### MoE (Mixture of Experts)

MoE **does NOT reduce KV-cache.** The formula is identical to dense:

    KV_MoE = same as the underlying attention formula for that architecture

MoE only reduces compute and weight memory. KV tensors store full hidden representations.

### KV Quantization

| Precision | B | Relative Size |
|-----------|---|--------------|
| BF16/FP16 | 2 | 1.0× |
| FP8/INT8 | 1 | 0.5× |
| INT4 | 0.5 | 0.25× |

---

## 2. Architecture Specifications from HuggingFace config.json

### Qwen 3.6-27B (Dense, Gated DeltaNet + Gated Attention)

| Parameter | Value |
|-----------|-------|
| model_type | qwen3_5 |
| num_hidden_layers (L) | 64 |
| hidden_size | 5,120 |
| num_attention_heads (H_q) | 24 |
| num_key_value_heads (H_kv) | 4 (for full-attention layers) |
| head_dim (D, full attn) | 256 |
| max_position_embeddings | 262,144 |
| intermediate_size | 17,408 |
| vocab_size | 248,320 |
| **Attention pattern** | 3× Gated DeltaNet + 1× Gated Attention, repeating x16 |
| DeltaNet: linear_num_key_heads | 16 |
| DeltaNet: linear_key_head_dim | 128 |
| DeltaNet: linear_num_value_heads | 48 |
| DeltaNet: linear_value_head_dim | 128 |
| DeltaNet: conv_kernel_dim | 4 |
| attn_output_gate | True (output_gate_type: swish) |
| full_attention_interval | 4 |
| partial_rotary_factor | 0.25 |
| rope_theta | 10,000,000 |

### Qwen 3.6-35B-A3B (MoE, Gated DeltaNet + Gated Attention)

| Parameter | Value |
|-----------|-------|
| model_type | qwen3_5_moe |
| num_hidden_layers (L) | 40 |
| hidden_size | 2,048 |
| num_attention_heads (H_q) | 16 |
| num_key_value_heads (H_kv) | 2 (for full-attention layers) |
| head_dim (D, full attn) | 256 |
| max_position_embeddings | 262,144 |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 512 |
| shared_expert_intermediate_size | 512 |
| total params | ~35B |
| active params | ~3B (8 routed + 1 shared expert) |
| **Attention pattern** | 3× Gated DeltaNet + 1× Gated Attention, repeating x10 |
| DeltaNet: linear_num_key_heads | 16 |
| DeltaNet: linear_key_head_dim | 128 |
| DeltaNet: linear_num_value_heads | 32 |
| DeltaNet: linear_value_head_dim | 128 |
| full_attention_interval | 4 |
| partial_rotary_factor | 0.25 |
| rope_theta | 10,000,000 |

### Qwen 3.5-32B (Dense, predecessor to 3.6)

| Parameter | Value |
|-----------|-------|
| num_hidden_layers (L) | 64 |
| hidden_size | 5,120 |
| num_attention_heads (H_q) | 40 |
| num_key_value_heads (H_kv) | 8 |
| head_dim | 128 |
| max_position_embeddings | 131,072 |
| intermediate_size | 27,648 |
| max_window_layers | 64 |
| sliding_window | 131,072 (effectively no window — full attention) |
| **Attention** | Full GQA (no sliding window in practice) |

### Gemma 4-31B (Dense, Hybrid SWA + Global Attention)

| Parameter | Value |
|-----------|-------|
| model_type | gemma4 |
| num_hidden_layers (L) | 60 |
| hidden_size | 5,376 |
| num_attention_heads (H_q) | 32 |
| num_key_value_heads (H_kv, SWA layers) | 16 |
| num_global_key_value_heads (SWA→Global transition) | 4 |
| head_dim (D) | 256 |
| global_head_dim | 512 |
| max_position_embeddings | 262,144 |
| intermediate_size | 21,504 |
| sliding_window (W) | 1,024 |
| final_logit_softcapping | 30.0 |
| vocab_size | 262,144 |
| attention_k_eq_v | True |
| **Attention pattern** | 5× SWA + 1× Full, repeating x10 = 50 SWA + 10 Full |
| RoPE (full attn) | rope_theta=1M, rope_type=proportional, partial=0.25 |
| RoPE (SWA) | rope_theta=10K, rope_type=default |

### Gemma 4-26B-A4B (MoE, Hybrid SWA + Global Attention)

| Parameter | Value |
|-----------|-------|
| model_type | gemma4_text |
| num_hidden_layers (L) | 30 |
| hidden_size | 2,816 |
| num_attention_heads (H_q) | 16 |
| num_key_value_heads (H_kv) | 8 |
| head_dim (D) | 256 |
| global_head_dim | 512 |
| num_global_key_value_heads | 2 |
| max_position_embeddings | 262,144 |
| intermediate_size | 2,112 |
| moe_intermediate_size | 704 |
| num_experts | 128 |
| top_k_experts | 8 |
| sliding_window (W) | 1,024 |
| **Attention pattern** | 5× SWA + 1× Full, repeating x5 = 25 SWA + 5 Full |
| enable_moe_block | True |
| **Total params** | ~25.2B |
| **Active params** | ~3.8B |

### DeepSeek-V3 (MLA)

| Parameter | Value |
|-----------|-------|
| num_hidden_layers (L) | 61 |
| hidden_size | 7,168 |
| num_attention_heads | 128 |
| num_key_value_heads | 128 |
| total_head_dim | 128 |
| kv_lora_rank | 512 |
| qk_rope_head_dim | 64 |
| qk_nope_head_dim | 64 |
| v_head_dim | 128 |
| **Attention** | MLA (Multi-Latent Attention) |
| **Total params** | 671B |
| **Active params** | 37B |

---

## 3. KV-Cache Calculations (BF16, B=2)

### How Qwen 3.6 KV-Cache Works (CRITICAL)

Qwen 3.6 uses **Gated DeltaNet** (linear attention) for 3 out of every 4 layers.
Linear attention uses a **fixed-size state matrix**: per-layer-per-token state = H_QK × D_QK + H_V × D_V.
This is **NOT cached per-sequence-position** — it's a single accumulated state updated incrementally.

Only the **Gated Attention** layers (every 4th layer) store K and V per token.

**For Qwen 3.6-27B:**
- Full-attention layers: L/4 = 16 layers, each: 2 × H_kv(4) × D(256) × 2 bytes = 4,096 bytes/token
- Linear attention state: L - L/4 = 48 layers, each: (16×128 + 48×128) × 2 = 16,384 bytes (FIXED, no S scaling)
- KV per token per full-attn layer: 4,096 bytes
- Total KV-cache = 16 × 4,096 × S + 48 × 16,384 bytes
- Simplified: 65,536 × S + 786,432 bytes

**For Qwen 3.6-35B-A3B:**
- Full-attention layers: L/4 = 10 layers, each: 2 × H_kv(2) × D(256) × 2 = 2,048 bytes/token
- Linear attention state: L - L/4 = 30 layers, each: (16×128 + 32×128) × 2 = 12,288 bytes (FIXED)
- KV per token per full-attn layer: 2,048 bytes
- Total KV-cache = 10 × 2,048 × S + 30 × 12,288 bytes
- Simplified: 20,480 × S + 368,640 bytes

### Qwen 3.6-27B (Dense, Hybrid Linear + Full Attention)

Multi-base: KV = 65,536 × S + 786,432 bytes (linear layers are fixed state)

| Context | KV Cache (S term) | Linear state | Total | GiB |
|---------|-------------------|-------------:|------:|----:|
| 4K (4,096) | 268,435,456 | 786,432 | 269 MB | 0.25 |
| 16K (16,384) | 1,073,741,824 | 786,432 | 1.01 GB | 0.94 |
| 64K (65,536) | 4,294,967,296 | 786,432 | 4.00 GB | 3.72 |
| 128K (131,072) | 8,589,934,592 | 786,432 | 8.01 GB | 7.46 |
| 256K (262,144) | 17,179,869,184 | 786,432 | 16.01 GB | 14.91 |

Note: Linear layers add 786,432 bytes (0.75 MiB) regardless of context. Growth is driven entirely by the 16 full-attention layers.

### Qwen 3.6-35B-A3B (MoE, Hybrid Linear + Full Attention)

Multi-base: KV = 20,480 × S + 368,640 bytes

| Context | KV Cache (S term) | Linear state | Total | GiB |
|---------|-------------------|-------------:|------:|----:|
| 4K (4,096) | 83,886,080 | 368,640 | 80.4 MB | 0.08 |
| 16K (16,384) | 335,544,320 | 368,640 | 319 MB | 0.30 |
| 64K (65,536) | 1,342,177,280 | 368,640 | 1.25 GB | 1.17 |
| 128K (131,072) | 2,684,354,560 | 368,640 | 2.50 GB | 2.33 |
| 256K (262,144) | 5,368,709,120 | 368,640 | 5.00 GB | 4.66 |

Note: Only 10 layers do standard KV-caching. This model has a remarkably small KV-cache despite 35B total params.

### Gemma 4-31B (Hybrid SWA + Global)

Pattern: 5 SWA (W=1024) + 1 Full → 50 SWA + 10 Full
Full/global layers use K=V unification (`attention_k_eq_v=True`) with separate head params:
  Full layers: 10 × 1 × 4 × 512 × 2 = 40,960 bytes/token (K=V, global_kv_heads=4, global_head_dim=512)
  SWA layers: 50 × 2 × 16 × 256 × 2 × min(S, 1024) = 819,200 × min(S, 1024) (standard K+V)

| Context | Full layers contrib | SWA layers contrib | Total | GiB |
|---------|--------------------:|-------------------:|------:|----:|
| 4K (4,096) | 167,772,160 | 838,860,800 | 1,006,632,960 | 0.94 |
| 16K (16,384) | 671,088,640 | 838,860,800 | 1,509,949,440 | 1.41 |
| 64K (65,536) | 2,684,354,560 | 838,860,800 | 3,523,215,360 | 3.28 |
| 128K (131,072) | 5,368,709,120 | 838,860,800 | 6,207,569,920 | 5.78 |
| 256K (262,144) | 10,737,418,240 | 838,860,800 | 11,576,279,040 | 10.78 |

Note: SWA layers are capped at W=1024. Full/global layers use K=V unification and only 4 KV heads (vs 16 for SWA). SWA adds fixed ~839 MB regardless of S>1K.

### Gemma 4-26B-A4B (MoE, Hybrid SWA + Global)

Pattern: 5 SWA (W=1024) + 1 Full → 25 SWA + 5 Full
Full/global layers use K=V unification with separate head params:
  Full layers: 5 × 1 × 2 × 512 × 2 = 10,240 bytes/token (K=V, global_kv_heads=2, global_head_dim=512)
  SWA layers: 25 × 2 × 8 × 256 × 2 × min(S, 1024) = 204,800 × min(S, 1024) (H_kv=8, standard K+V)

| Context | Full layers | SWA layers (capped) | Total | GiB |
|---------|------------:|--------------------:|------:|----:|
| 4K (4,096) | 41,943,040 | 209,715,200 | 251,658,240 | 0.23 |
| 16K (16,384) | 167,772,160 | 209,715,200 | 377,487,360 | 0.35 |
| 64K (65,536) | 671,088,640 | 209,715,200 | 880,803,840 | 0.82 |
| 128K (131,072) | 1,342,177,280 | 209,715,200 | 1,551,892,480 | 1.45 |
| 256K (262,144) | 2,684,354,560 | 209,715,200 | 2,894,069,760 | 2.70 |

### DeepSeek-V3 (MLA)

KV_MLA = 61 × S × (512 + 64) × 2 = 61 × S × 1,152 = 70,272 × S

| Context | KV Cache | GiB |
|---------|---------:|----:|
| 4K (4,096) | 287,834,112 | 0.27 |
| 16K (16,384) | 1,151,336,448 | 1.07 |
| 64K (65,536) | 4,605,345,792 | 4.29 |
| 128K (131,072) | 9,210,691,584 | 8.58 |
| 256K (262,144) | 18,421,383,168 | 17.15 |

### Qwen 3.5-32B (Dense, Full GQA)

For reference: L=64, H_kv=8, D=128 → 2×8×128×2 = 4,096 bytes/token/layer
Across 64 layers: 262,144 bytes/token

| Context | KV Cache | GiB |
|---------|---------:|----:|
| 4K | 1,073,741,824 | 1.00 |
| 16K | 4,294,967,296 | 4.00 |
| 128K | 34,359,738,368 | 32.00 |
| 256K | 68,719,476,736 | 64.00 |

---

## 4. 128K Context Comparison (BF16, ranked smallest first)

| Rank | Model | Architecture | KV @ 128K | GiB |
|------|-------|-------------|-----------|----:|
| 1 | Gemma 4-26B-A4B | 25 SWA + 5 Full (K=V), MoE | 10,240×S + capped SWA | 1.45 |
| 2 | Qwen 3.6-35B-A3B | Linear attn (30 layers) + Full attn (10 layers), MoE | 20,480×S + fixed | 2.33 |
| 3 | Gemma 4-31B | 50 SWA + 10 Full (K=V), Dense | 40,960×S + capped SWA | 5.78 |
| 4 | Qwen 3.6-27B | Linear attn (48 layers) + Full attn (16 layers), Dense | 65,536×S + fixed | 7.46 |
| 5 | DeepSeek-V3 (671B!) | MLA compression | 70,272×S | 8.58 |
| 6 | Qwen 3.5-32B | Full GQA, 64 layers | 262,144×S | 32.00 |

---

## 5. INT4 Quantization at 128K (0.25× BF16)

| Model | BF16 @ 128K | INT4 @ 128K | Savings |
|-------|------------:|------------:|--------:|
| Gemma 4-26B-A4B | 1.45 GiB | 0.36 GiB | 1.09 GiB |
| Qwen 3.6-35B-A3B | 2.33 GiB | 0.58 GiB | 1.75 GiB |
| Gemma 4-31B | 5.78 GiB | 1.45 GiB | 4.33 GiB |
| Qwen 3.6-27B | 7.46 GiB | 1.86 GiB | 5.60 GiB |
| DeepSeek-V3 (671B) | 8.58 GiB | 2.15 GiB | 6.43 GiB |
| Qwen 3.5-32B | 32.00 GiB | 8.00 GiB | 24.00 GiB |

---

## 6. Total VRAM Estimation (128K, BF16)

Formula: VRAM ≈ Weights_bytes + KV_cache + Activations(0.3-0.5 GB) + CUDA_overhead(1-2 GB)

| Model | Weights (BF16) | KV @ 128K | Overhead | Total | Hardware |
|-------|---------------:|----------:|---------:|------:|----------|
| Gemma 4-26B-A4B | ~50 GB (MoE) | 1.45 GiB | ~1 GB | ~52.5 GiB | 1× A100 80GB |
| Qwen 3.6-35B-A3B | ~69 GB (weights, MoE) | 2.33 GiB | ~1 GB | ~72.5 GiB | 1× A100 80GB / 2× RTX 4090 |
| Gemma 4-31B | ~61 GB | 5.78 GiB | ~1.5 GB | ~68.3 GiB | 1× A100 80GB |
| Qwen 3.6-27B | ~54 GB | 7.46 GiB | ~1.5 GB | ~63.0 GiB | 1× A100 80GB |
| DeepSeek-V3 | ~1,342 GB (sharded) | 8.58 GiB | — | Weight-bound | KV is ~0.6% of weight size |
| Qwen 3.5-32B | ~64 GB | 32.00 GiB | ~1.5 GB | ~97.5 GiB | 2× A100 80GB |

---

## 7. Critical Finding: Linear Attention vs Sliding Window vs Full

This is the key insight for models released in 2026. There are now **three fundamentally different** ways to avoid O(S²) scaling:

**1. Sliding Window Attention** (Gemma 4): KV-cache still grows O(S) but only for full-attention layers. SWA layers cap at W tokens. Net growth rate = (L_full / L) × full_rate.

**2. Linear Attention / Gated DeltaNet** (Qwen 3.6): Linear attention layers maintain a **fixed-size state matrix** that's O(1) with respect to sequence length. Only the interleaved full-attention layers contribute O(S) cache. Net growth rate = (1 / full_attn_interval) × full_rate.

**3. MLA** (DeepSeek-V3): Compresses KV per token via low-rank projections. Still O(S) but with a dramatically reduced coefficient (~57× smaller per layer).

**Comparison at S → ∞:**

The "effective rate" below compares each model's per-token growth to what a naive full-attention model with the same head dims would use. Note that Gemma 4's effective rate is further reduced by its K=V unification on global layers.

| Architecture | Asymptotic KV growth | Effective rate |
|--------------|---------------------|----------------|
| Full attention | O(S) | 1.0× |
| Qwen 3.6 (3:1 Linear:Full) | O(S) | 1/4 = 25% |
| Qwen 3.6-27B (3:1, 64L) | O(S) | 1/4 = 25% |
| Qwen 3.6-35B-A3B (3:1, 40L) | O(S) | 1/4 = 25% |
| Gemma 4 (5:1 SWA:Full, K=V) | O(S) | 1/6 × ~0.25 (K=V reduces full-layer rate) |
| DeepSeek-V3 (MLA) | O(S) | ~1.7% (vs full attn equivalent) |

Note: Gemma 4's global/full layers use K=V unification and far fewer KV heads than the SWA layers. At long context (S >> W), only the 1/6 global layers grow, and they do so at a fraction of the per-head rate due to K=V sharing. Qwen 3.6's linear layers have ZERO S-scaling (O(1) constants). At short context, Qwen 3.6 has slightly higher fixed-state overhead but grows slower because the full-attention layers use aggressive GQA (Q=24, KV=4).

---

## 8. Practical Guidance

### What to measure when comparing models

1. Don't just look at param count — count **KV-cache bytes per token** = 2 × Σ_layer(H_kv_i × D_i)
2. Check if the model uses **linear attention**, **SWA**, or **MLA** — these fundamentally change cache scaling
3. Count **how many layers actually cache K and V per token** — many 2026 models only cache on 1/4 to 1/6 of layers
4. For MoE models: MoE reduces **weight memory** and **compute**, but NOT KV-cache

### Best models for long-context local inference

- **Gemma 4-26B-A4B**: Smallest KV-cache (1.45 GiB @ 128K BF16), 3.8B active compute. Best KV efficiency thanks to K=V unification and SWA capping.
- **Qwen 3.6-35B-A3B**: Very small KV-cache (2.33 GiB @ 128K BF16), 3B active compute, 35B in memory. Excellent KV efficiency.
- **Gemma 4-31B**: Moderate KV-cache (5.78 GiB @ 128K), dense, benefits from K=V unification on global layers.
- **Qwen 3.6-27B**: Dense model with only 7.46 GiB @ 128K. Excellent for a 27B dense model.

---

## 9. References

- Raschka, S. "KV Cache Calculations." https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/
- Raschka, S. "LLM Architecture Gallery." https://sebastianraschka.com/llm-architecture-gallery/
- Raschka, S. "Visual Attention Variants." https://magazine.sebastianraschka.com/p/visual-attention-variants
- Qwen Team. "Qwen3.6-27B" (HuggingFace config.json)
- Qwen Team. "Qwen3.6-35B-A3B" (HuggingFace config.json)
- Google DeepMind. "Gemma 4-31B" (HuggingFace config.json, gemma-4-31b-it)
- Google DeepMind. "Gemma 4-26B-A4B" (HuggingFace config.json, gemma-4-26b-a4b-it)
- DeepSeek-AI. "DeepSeek-V3" (arXiv:2412.19437)
- Thomas Heliez. "Gemma 4 Explained." https://thomasthelliez.com/blog/gemma-4-explained-architecture-benchmarks-use-cases/
- InsiderLLM. "Qwen 3.6 Complete Guide." https://insiderllm.com/guides/qwen-3-6-local-ai-guide/

---

*Document generated: 2026-05-13 · Math corrected: 2026-05-15
Configs fetched from: HuggingFace raw config.json endpoints
Key finding: Gemma 4's K=V unification on global layers makes it the most KV-efficient architecture at long context. Qwen 3.6's hybrid DeltaNet achieves 25% growth rate. DeepSeek's MLA achieves ~1.7% per-layer rate but has 61 layers. All three families dramatically outperform full-attention models like Qwen 3.5-32B.*
