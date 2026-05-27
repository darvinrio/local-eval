# KV Cache / Token (bf16) | Sebastian Raschka, PhD

## Source

https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/

## What it means

This page explains the `KV cache / token (bf16)` number shown in the gallery fact sheets. It is a simple logical estimate of how much cache state one new token adds at inference time (for batch size 1), using the common bf16 ([16-bit bfloat](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)) precision format (16 bits =2 bytes per element).

The gallery does not try to report framework-specific runtime memory, allocator padding, or kernel overhead here. It only reports the underlying cache geometry implied by the published model architecture.

How much cache grows when one token is appended during decoding

Units

Binary units, so `1 KiB = 1024 bytes`

Important caveat

This is logical cache size, not measured serving memory

### Baseline Definition[](#baseline-definition)

The gallery uses these conventions:

*   batch size = `1`
*   dtype = `bf16`
*   bytes per cached element = `2`
*   the metric is per generated token, not full-context footprint

If you want total cache at a given sequence length later, you can multiply the per-token number by the number of stored tokens, adjusting for any local-window or sparse-retention rules.

### Standard Attention: MHA, GQA, and MQA[](#standard-attention-mha-gqa-and-mqa)

For standard attention layers, each new token appends one new _key_ vector and one new _value_ vector for every KV head:

```
bytes_per_layer_per_token
= 2 × num_kv_heads × head_dim × 2
= 4 × num_kv_heads × head_dim

```


Where:

*   the first `2` is for `K` and `V`
*   `num_kv_heads = num_attention_heads` for MHA
*   `num_kv_heads < num_attention_heads` for GQA
*   `num_kv_heads = 1` for MQA

For a whole model:

```
bytes_per_model_per_token
= sum(bytes_per_layer_per_token across all cache-growing attention layers)

```


Sliding-window attention uses the same per-token formula as its underlying MHA/GQA/MQA layer. It changes how much cache is retained at long sequence lengths, but not how much one stored token costs.

Some architectures reduce this further by explicitly sharing or unifying keys and values in certain layers. In that case, the cache only stores one tensor instead of separate `K` and `V` tensors:

```
bytes_per_layer_per_token
= num_kv_heads × head_dim × 2
= 2 × num_kv_heads × head_dim

```


Gemma 4’s full-attention layers are one example. Those layers also use a separate `global_head_dim`, so the total cache cost is the sum of:

*   regular sliding-window layers using the standard `K` + `V` formula
*   global layers using the unified `K=V` formula
    

### MLA[](#mla)

For DeepSeek-style multi-head latent attention (MLA), the gallery uses the compressed latent cache representation rather than an expanded implementation fallback.

Per MLA layer:

```
bytes_per_layer_per_token
= (kv_lora_rank + qk_rope_head_dim) × 2

```


For a whole model:

```
bytes_per_model_per_token
= num_mla_layers × (kv_lora_rank + qk_rope_head_dim) × 2

```


This is why MLA-based models can have a much smaller cache footprint than similarly sized GQA models.

### Hybrids and Recurrent Models[](#hybrids-and-recurrent-models)

Hybrid models are handled by counting only the layers that actually append to a growing KV cache.

*   Qwen3-Next and Qwen3.5: only the full attention layers count
*   Kimi Linear and Ling 2.5/2.6: only the MLA layers in the main decoder stack count; Ling 2.6’s optional MTP path can add a small extra MLA cache when used
*   Nemotron hybrids: only the explicit GQA layers count
*   DeltaNet, Lightning Attention, Mamba-2, and xLSTM layers contribute `0 B/token` to a growing KV cache because they use fixed recurrent state instead

So the gallery value for a hybrid is:

```
sum(per-layer cache growth over the cache-growing layers only)

```


### Concrete Examples[](#concrete-examples)

`Qwen3 8B`

```
36 layers × 8 KV heads × 128 head_dim × 4
= 147,456 bytes
= 144 KiB

```


`DeepSeek V3`

```
61 layers × (512 kv_lora_rank + 64 qk_rope_head_dim) × 2
= 70,272 bytes
= 68.6 KiB

```


`Qwen3 Next 80B-A3B`

```
12 full-attention layers × 2 KV heads × 256 head_dim × 4
= 24,576 bytes
= 24 KiB

```


`Gemma 4 26B-A4B`

```
25 sliding-window layers × 8 KV heads × 256 head_dim × 4
+ 5 global layers × 2 KV heads × 512 global_head_dim × 2
= 204,800 + 10,240 bytes
= 215,040 bytes
= 210 KiB

```


`Gemma 4 31B`

```
50 sliding-window layers × 16 KV heads × 256 head_dim × 4
+ 10 global layers × 4 KV heads × 512 global_head_dim × 2
= 819,200 + 40,960 bytes
= 860,160 bytes
= 840 KiB

```


The key point in both Gemma 4 examples is that the global layers use unified `K=V`, so those layers contribute only one cached tensor instead of separate key and value tensors.

`xLSTM 7B`

### Qualitative Bands Used In The Gallery[](#qualitative-bands-used-in-the-gallery)

For quick scanning, the gallery and diff tool also attach a rough qualitative label to the numeric `KV cache / token (bf16)` value:

*   `0 B` -> `No cache`
*   `> 0` and `<= 24 KiB` -> `Very low`
*   `> 24 KiB` and `<= 72 KiB` -> `Low`
*   `> 72 KiB` and `<= 160 KiB` -> `Moderate`
*   `> 160 KiB` and `<= 300 KiB` -> `High`
*   `> 300 KiB` -> `Very high`

These bands are only meant as a quick orientation aid. The actual numeric value remains the more important quantity.