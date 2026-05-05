# Local Evals

Evaluating local models

## models 

* `unsloth/gemma-4-26B-A4B-it`
* `unsloth/gemma-4-E4B-it`
* `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2`
* `leonsarmiento/Qwen3.6-27B-3bit-mlx`
* `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit`
* `mlx-community/Qwen3.5-9B-MLX-4bit`

## run mlx_lm

```sh
mlx_lm.server \
  --model Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2 \
  --port 8888
```

## run unsloth studio

```sh
unsloth studio -H 0.0.0.0 -p 8888
```

## run qwen coder llama-server

```sh
llama-server \                                    
  --model ./Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf \     
  --n-gpu-layers 99 \                                      
  -ot ".ffn_.*_exps.*=CPU" \
  --ctx-size 65536 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --parallel 1 \
  --no-warmup \
  --flash-attn on \
  --temp 0.7 --min-p 0.01 --top-p 0.80 --top-k 20 \
  --repeat-penalty 1.05 \
  --jinja \
  --host 0.0.0.0 --port 8888
```
