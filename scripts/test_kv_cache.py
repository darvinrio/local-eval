from utils.kv_cache.adapters import get_adapter, Gemma4Adapter, Qwen36HybridAdapter
from utils.kv_cache import estimate_kv_cache

print("Testing Gemma4 26B-A4B...")
# Gemma4 26B-A4B, bf16, num_kv_heads=8, head_dim=256
gemma4_config = {
    "model_type": "gemma4",
    "layer_types": ["full_attention"] * 5 + ["sliding_attention"] * 25,
    "num_key_value_heads": 8,
    "head_dim": 256,
    "sliding_window": 1024,
}
estimate = estimate_kv_cache(gemma4_config, kv_quant_bits=16)
print(f"Gemma4 128K: {estimate.estimate_gb(131072):.2f} GiB")

print("Testing Qwen3.6 35B-A3B...")
qwen36_moe_config = {
    "model_type": "qwen3_5_moe",
    "layer_types": ["full_attention"] * 10 + ["linear_attention"] * 30,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
}
estimate = estimate_kv_cache(qwen36_moe_config, kv_quant_bits=16)
print(f"Qwen3.6 35B-A3B 128K: {estimate.estimate_gb(131072):.2f} GiB")

print("Testing Qwen3.6 27B Dense...")
qwen36_dense_config = {
    "model_type": "qwen3_5",
    "layer_types": ["full_attention"] * 16 + ["linear_attention"] * 48,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
}
estimate = estimate_kv_cache(qwen36_dense_config, kv_quant_bits=16)
print(f"Qwen3.6 27B Dense 128K: {estimate.estimate_gb(131072):.2f} GiB")
