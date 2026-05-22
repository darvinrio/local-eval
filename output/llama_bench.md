# Benchmarking on Llama-Bench

## qwen35moe 35B.A3B IQ4_XS
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |          pp8192 |        571.01 ± 3.17 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |         pp32768 |        438.73 ± 0.48 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |         pp40000 |        398.29 ± 9.41 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |         pp50000 |        374.92 ± 1.86 |

| model                          |       size |     params | backend    | threads | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |   q4_1 |   q4_1 |  1 |         pp65536 |        272.52 ± 0.26 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |   q4_1 |   q4_1 |  1 |          tg1024 |         40.74 ± 1.40 |

| model                          |       size |     params | backend    | threads | n_ubatch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |     2048 |   q8_0 |   q8_0 |  1 |         pp65536 |        275.57 ± 0.49 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |     2048 |   q8_0 |   q8_0 |  1 |           tg128 |         41.63 ± 0.08 |

| model                          |       size |     params | backend    | threads | n_ubatch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |     2048 |   q4_1 |   q4_1 |  1 |        pp131072 |        184.08 ± 0.50 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | BLAS,MTL   |       8 |     2048 |   q4_1 |   q4_1 |  1 |           tg128 |         41.64 ± 0.04 |

## qwen35 27B Q4_K
| model                          |       size |     params | backend    | threads | n_ubatch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35 27B Q4_K - Small        |  14.76 GiB |    26.90 B | BLAS,MTL   |       8 |     2048 |   q4_1 |   q4_1 |  1 |          pp4096 |         97.49 ± 1.24 |
| qwen35 27B Q4_K - Small        |  14.76 GiB |    26.90 B | BLAS,MTL   |       8 |     2048 |   q4_1 |   q4_1 |  1 |          tg1024 |         11.56 ± 0.07 |

```csv
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,use_mmap,use_direct_io,embeddings,no_op_offload,no_host,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
"d05fe1d7d","9010","Accelerate, Apple M4 Pro","Apple M4 Pro","BLAS,MTL","/Users/darvin/models/Qwen3.6-27B-Q4_K_S.gguf","qwen35 27B Q4_K - Small","15845165056","26895998464","2048","2048","8","0x0","0","50","q4_1","q4_1","99","0","layer","0","0","1","auto","0.00","none","1","0","0","0","0","0","0","4096","0","0","2026-05-20T18:44:36Z","42102228666","601071207","97.302719","1.375385"
"d05fe1d7d","9010","Accelerate, Apple M4 Pro","Apple M4 Pro","BLAS,MTL","/Users/darvin/models/Qwen3.6-27B-Q4_K_S.gguf","qwen35 27B Q4_K - Small","15845165056","26895998464","2048","2048","8","0x0","0","50","q4_1","q4_1","99","0","layer","0","0","1","auto","0.00","none","1","0","0","0","0","0","0","8192","0","0","2026-05-20T18:48:49Z","87789513341","3900236458","93.327482","1.250980"
"d05fe1d7d","9010","Accelerate, Apple M4 Pro","Apple M4 Pro","BLAS,MTL","/Users/darvin/models/Qwen3.6-27B-Q4_K_S.gguf","qwen35 27B Q4_K - Small","15845165056","26895998464","2048","2048","8","0x0","0","50","q4_1","q4_1","99","0","layer","0","0","1","auto","0.00","none","1","0","0","0","0","0","0","16384","0","0","2026-05-20T18:57:34Z","182429410483","309064082","89.810286","0.152128"
"d05fe1d7d","9010","Accelerate, Apple M4 Pro","Apple M4 Pro","BLAS,MTL","/Users/darvin/models/Qwen3.6-27B-Q4_K_S.gguf","qwen35 27B Q4_K - Small","15845165056","26895998464","2048","2048","8","0x0","0","50","q4_1","q4_1","99","0","layer","0","0","1","auto","0.00","none","1","0","0","0","0","0","0","32768","0","0","2026-05-20T19:15:48Z","405975440383","4193541514","80.715709","0.384351"
"d05fe1d7d","9010","Accelerate, Apple M4 Pro","Apple M4 Pro","BLAS,MTL","/Users/darvin/models/Qwen3.6-27B-Q4_K_S.gguf","qwen35 27B Q4_K - Small","15845165056","26895998464","2048","2048","8","0x0","0","50","q4_1","q4_1","99","0","layer","0","0","1","auto","0.00","none","1","0","0","0","0","0","0","65536","0","0","2026-05-20T19:56:25Z","972347175941","343748495","67.399801","0.023816"
"d05fe1d7d","9010","Accelerate, Apple M4 Pro","Apple M4 Pro","BLAS,MTL","/Users/darvin/models/Qwen3.6-27B-Q4_K_S.gguf","qwen35 27B Q4_K - Small","15845165056","26895998464","2048","2048","8","0x0","0","50","q4_1","q4_1","99","0","layer","0","0","1","auto","0.00","none","1","0","0","0","0","0","0","4096","0","0","2026-05-22T13:18:29Z","42216608566","834438754","97.053144","1.879475"
```
