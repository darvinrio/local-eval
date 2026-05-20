oMLX - LLM inference, optimized for your Mac
https://github.com/jundot/omlx
Benchmark Model: Qwen3.6-35B-A3B-TurboQuant-MLX-3bit
================================================================================

```sh
Single Request Results
--------------------------------------------------------------------------------
Test                TTFT(ms)    TPOT(ms)        pp TPS        tg TPS      E2E(s)    Throughput    Peak Mem
pp8192/tg128         11216.1       12.88   730.4 tok/s    78.2 tok/s      12.852   647.4 tok/s    16.36 GB
pp16384/tg128        24224.2       15.08   676.3 tok/s    66.8 tok/s      26.140   631.7 tok/s    16.98 GB
pp32768/tg128        56851.3       18.02   576.4 tok/s    55.9 tok/s      59.140   556.2 tok/s    18.32 GB
pp65536/tg128       151254.4       23.77   433.3 tok/s    42.4 tok/s     154.273   425.6 tok/s    20.92 GB
```
