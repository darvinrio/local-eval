from mlx_lm import generate, load

model, tokenizer = load("Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2")

prompt = "Write a story about Einstein"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
