# Choosing a Local setup 

A write-up where we analyze and choose an appropriate local setup

[] - Understand your system
  - How much free ram can you get
  - Choices of inference engines
[] - Model Choices
  - Model preferences
  - Quantization to go with 
  - Context size and Output/Input token counts
  - Dense or MoE - MoE activate few layers and thus can give faster tps
  - Can I run multiple inferences for agents ?
[] - Design Eval or Benchmark
  - List out your local model use case
  - Define Benchmarks that can standardize and test multiple setups
  - Because, you might have to compare different quants of different models, hence different results and tps
  - ttf - time to first token might be more important for code completion within editors
