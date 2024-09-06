# llama2.ts
PoC and educational inference for [Llama2]-like Transformer models (phi, gemma, tinyllama, etc) in TypeScript

Based on Oleksandr Nikitin's llama2.ts: https://github.com/wizzard0/llama2.ts

Based on the Andrej Karpathy's [llama2.c].

### Features
- WASM acceleration
- JIT Quantization (Q8_0)
- LoRA Adaptation
- Safetensors loader
- BPE tokenizer.json loader 

### Usage

deno:
```sh
git clone https://huggingface.co/nickypro/tinyllama-15M
cd src/wasm && make all    # requires emcc
deno run --allow-all llama2.ts ./tinyllama-15M -s 1 -t 0 -i "Once upon a time"
```

Arguments:
- `-i <string>` - initial prompt
- `-t <float>` - temperature (0..1, 0 = deterministic argmax)
- `-s <int>` - random seed
- `-n <int>` - number of tokens to generate (0..256, default 256)
- `-p <float>` - p value for nucleus sampling, default 0.9
- `-a <string>` - LoRA Adapter path (adapter.safetensors folder)

[llama2.ts]: https://github.com/wizzard0/llama2.ts
[t348]: https://github.com/wizzard0/t348-loader
[TinyStories]: https://arxiv.org/abs/2305.07759
[llama2.c]: https://github.com/karpathy/llama2.c
[Llama2]: https://ai.meta.com/llama/
[llama2.js]: https://github.com/epicure/llama2.js
