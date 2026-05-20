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
