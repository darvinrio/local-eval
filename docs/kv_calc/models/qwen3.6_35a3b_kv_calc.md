Using the Hugging Face config, the **exact KV cache size at 256K context is 5,368,709,120 bytes**, which is 5.00 GiB for binary units or 5.37 GB for decimal units, assuming the standard bf16/fp16 KV cache element size of 2 bytes.  This is small for a 35B-class model because Qwen3.6-35B-A3B uses a hybrid stack where only every 4th layer is a full gated-attention layer, while the other layers are Gated DeltaNet linear-attention blocks that do not store a growing per-token Transformer-style KV cache. [news.ycombinator](https://news.ycombinator.com/item?id=47871242)

## Config math

From the Hugging Face config, the relevant fields are: `num_hidden_layers = 40`, `full_attention_interval = 4`, `num_key_value_heads = 2`, `head_dim = 256`, and `max_position_embeddings = 262144`.  That implies only \(40 / 4 = 10\) full-attention layers contribute to the standard growing KV cache. [huggingface](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)

Per token, the KV cache cost is:
- K: \(2 \text{ KV heads} \times 256 \text{ dim} \times 10 \text{ layers} \times 2 \text{ bytes} = 10{,}240\) bytes [huggingface](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
- V: same \(= 10{,}240\) bytes [huggingface](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
- Total per token: \(20{,}480\) bytes [huggingface](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)

At 262,144 tokens, total KV cache is:
\[
262{,}144 \times 20{,}480 = 5{,}368{,}709{,}120 \text{ bytes}
\]
 [huggingface](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)

## Why it is reduced

The reduction does **not** come from compressing normal attention K/V with extra linear projection tricks inside those 10 full-attention layers, at least not from the public HF config fields you pointed to.  The main savings come from architecture: 30 of the 40 layers are linear-attention/Gated DeltaNet layers, so they avoid the usual \(O(n)\) per-token KV growth that a standard Transformer attention layer would incur. [vast](https://vast.ai/model/qwen36-35b-a3b)

That means the honest interpretation is: the model’s long-context memory advantage is primarily due to having only 10 cache-bearing attention layers, not because standard KV tensors are further compressed by some hidden HF-config KV packing scheme.  Claims about extra KV compression beyond this should be treated as unverified unless backed by implementation code or runtime logs. [huggingface](https://huggingface.co/docs/transformers/en/kv_cache)

## Cross-check

A real llama.cpp runtime log reports for 262,144 context and 10 layers: `K (f16): 2560.00 MiB`, `V (f16): 2560.00 MiB`, total `5120.00 MiB`.  That matches the config-derived total exactly: \(2560 + 2560 = 5120\) MiB, which is \(5 \times 1024^3 = 5{,}368{,}709{,}120\) bytes. [news.ycombinator](https://news.ycombinator.com/item?id=47871242)

## Practical verdict

On your Apple Silicon workflow, this is the number that matters: **5.0 GiB just for KV cache at full 256K context**, before weights, activations, runtime overhead, and any speculative decoding buffers.  So if you only have 10–16 GB realistically free, 256K context is technically possible only with very aggressive weight quantization and a lean runtime; otherwise it is a bad fit on a 24 GB unified-memory machine once the rest of the system is alive. [news.ycombinator](https://news.ycombinator.com/item?id=47871242)

Uncertainty tier: **1, confirmed from config and runtime log**. [news.ycombinator](https://news.ycombinator.com/item?id=47871242)

If you want, I can also compute the **total memory budget** for this model on your M4 Pro for specific quants like Q4, Q5, or MXFP4.
