# Local Evals

Evaluating local models

## models

* `unsloth/gemma-4-26B-A4B-it`
* `unsloth/gemma-4-E4B-it`
* `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2`
* `leonsarmiento/Qwen3.6-27B-3bit-mlx`
* `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit`
* `mlx-community/Qwen3.5-9B-MLX-4bit`
* `unsloth/Qwen3.6-27B-MTP-GGUF', variant=UD-Q3_K_XL`
* `unsloth/Qwen3.6-35B-A3B-GGUF:Q3_K_S`

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
  --model Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \     
  --ctx-size 65536 \
  --cache-type-k q4_1 \
  --cache-type-v q4_1 \
  --parallel 1 \
  --temp 0.7 --min-p 0.01 --top-p 0.80 --top-k 20 \
  --repeat-penalty 1.05 \
  --jinja \
  --host 0.0.0.0 --port 8888
```

```sh
llama-server \
  --model ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/blobs/649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3 \
  --jinja \
  -c 65536 \
  -ctk q4_1 -ctv q4_1 \
  --host 0.0.0.0 --port 8888
```

## run model via unsloth cli

```sh
unsloth run --model unsloth/Qwen3.6-35B-A3B-GGUF:Q3_K_S -c 131072 -ctk q4_1 -ctv q4_1
```

## run mtp model via unsloth cli

```sh
unsloth run \
  --model unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q3_K_XL \
  -c 131072 -ctk q4_1 -ctv q4_1 \
  --spec-type draft-mtp --spec-draft-n-max 6 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --api-key ha
```

## helpful commands

### mac iogpu

```sh
sudo sysctl ogpu.wired_limit_mb=20480 
```

### python

```sh
# check ruff issues
ruff check
# fix ruff issues
ruff check --fix  
# format ruff
ruff format 
# check types
ty check
# run all checks
pre-commit run --all-files
```

```sh
# add repo to python path
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### gitHub

```bash
# Delete local merged branches
git branch --merged | grep -v '\*' | xargs -n 1 git branch -d

# Prune origin deleted branches
git remote prune origin
```
