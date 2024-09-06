// Llama2 transformer model inference in one TypeScript file.
// by Luca Elin Haneklau, 2024 (MIT licensed).
// Based on Oleksandr Nikitin's llama2.ts: https://github.com/wizzard0/llama2.ts
// Based on Andrej Karpathy's llama2.c: https://github.com/karpathy/llama2.c

// deno-lint-ignore-file prefer-const

import { writeAllSync } from "https://deno.land/std@0.224.0/io/mod.ts";
import {
  accum,
  elemmul,
  geglu,
  gelu,
  matmul,
  quantize,
  rmsnorm,
  silu,
} from "./kernels.ts";
import { qmatmul } from "./kernels_wasm.ts";
import { lora, mutlihead_attention, rope } from "./transformer.ts";
import { decode, encode, readHFRepoTokenizer } from "./sentencepiece.ts";
import { sample } from "./sampler.ts";
import { set_seed } from "./rng.ts";
//import { type F32Tensor, type Q8Tensor } from "./types.ts";
import { WasmF32Tensor, WasmQ8Tensor } from "./types_wasm.ts";
import { JSQ8Tensor } from "./types.ts";
import {
  AdapterConfig,
  AdapterWeights,
  Config,
  QuantizedTransformerWeights,
  readAdapter,
  readModel,
} from "./model.ts";

type F32Tensor = WasmF32Tensor;
type Q8Tensor = WasmQ8Tensor;

interface RunState {
  x: F32Tensor;
  x_q: Q8Tensor;
  xb: F32Tensor;
  xb2: F32Tensor;
  hb: F32Tensor;
  hb_q: Q8Tensor;
  hb2: F32Tensor;
  q: F32Tensor;
  k: F32Tensor;
  v: F32Tensor;
  att: F32Tensor;
  logits: F32Tensor;
  key_cache: F32Tensor;
  value_cache: F32Tensor;
}

interface AdapterRunState {
  q_r: F32Tensor;
  k_r: F32Tensor;
  v_r: F32Tensor;
  o_r: F32Tensor;
  gate_r: F32Tensor;
  down_r: F32Tensor;
  up_r: F32Tensor;
}

function newRunState(config: Config): RunState {
  const kv_dim = (config.hidden_size * config.n_kv_heads) / config.n_heads;
  const s = {} as RunState;
  s.x = WasmF32Tensor.allocate(config.hidden_size);
  s.x_q = WasmQ8Tensor.allocate(config.hidden_size, config.group_size);
  s.xb = WasmF32Tensor.allocate(config.hidden_size);
  s.xb2 = WasmF32Tensor.allocate(config.hidden_size);
  s.hb = WasmF32Tensor.allocate(config.intermediate_size);
  s.hb_q = WasmQ8Tensor.allocate(config.intermediate_size, config.group_size);
  s.hb2 = WasmF32Tensor.allocate(config.intermediate_size);
  s.q = WasmF32Tensor.allocate(config.hidden_size);
  s.k = WasmF32Tensor.allocate(kv_dim);
  s.v = WasmF32Tensor.allocate(kv_dim);
  s.att = WasmF32Tensor.allocate(config.n_heads * config.seq_len);
  s.logits = WasmF32Tensor.allocate(config.vocab_size);
  s.key_cache = WasmF32Tensor.allocate(
    config.n_layers * config.seq_len * kv_dim,
  );
  s.value_cache = WasmF32Tensor.allocate(
    config.n_layers * config.seq_len * kv_dim,
  );
  return s;
}
function newAdapterState(config: AdapterConfig): AdapterRunState {
  const s = {} as AdapterRunState;
  s.q_r = WasmF32Tensor.allocate(config.rank);
  s.k_r = WasmF32Tensor.allocate(config.rank);
  s.v_r = WasmF32Tensor.allocate(config.rank);
  s.down_r = WasmF32Tensor.allocate(config.rank);
  s.o_r = WasmF32Tensor.allocate(config.rank);
  s.gate_r = WasmF32Tensor.allocate(config.rank);
  s.up_r = WasmF32Tensor.allocate(config.rank);
  return s;
}

function forward(
  token: number,
  pos: number,
  conf: Config,
  state: RunState,
  weights: QuantizedTransformerWeights,
  adapter_config?: AdapterConfig,
  adapter_state?: AdapterRunState,
  adapter_weights?: AdapterWeights,
): void {
  const kv_dim = (conf.hidden_size * conf.n_kv_heads) / conf.n_heads;

  const hidden_dim = conf.intermediate_size;
  const head_size = conf.hidden_size / conf.n_heads;

  const adapter_scale = adapter_config
    ? adapter_config.rank / adapter_config.alpha
    : 0;

  // copy the token embedding into x
  if (weights.embed_tokens instanceof WasmQ8Tensor) {
    const embeddingQuant = weights.embed_tokens.subarray(
      token * conf.hidden_size,
      token * conf.hidden_size + conf.hidden_size,
    );
    const embedding = embeddingQuant.dequantize();
    for (let i = 0; i < conf.hidden_size; i++) {
      state.x.array[i] = embedding[i];
    }
  } else {
    const embedding = weights.embed_tokens.subarray(
      token * conf.hidden_size,
      token * conf.hidden_size + conf.hidden_size,
    );
    for (let i = 0; i < conf.hidden_size; i++) {
      state.x.array[i] = embedding.array[i];
    }
  }

  const embedding_scaling_factor = conf.model_type === "gemma"
    ? Math.sqrt(conf.hidden_size)
    : 1;
  for (let i = 0; i < conf.hidden_size; i++) {
    state.x.array[i] = state.x.array[i] * embedding_scaling_factor;
  }

  //debugger;
  // forward all the layers
  for (let l = 0; l < conf.n_layers; l++) {
    //console.log("Layer %d", l + 1);

    // attention rmsnorm
    rmsnorm(
      state.xb,
      state.x,
      weights.input_layernorm[l],
      conf.hidden_size,
      conf.norm_eps,
    );

    // key and value point to the kv cache
    const kv_cache_layer_offset = l * conf.seq_len * kv_dim; // kv cache layer offset for convenience
    const kv_cache_layer_pos_offset = kv_cache_layer_offset + pos * kv_dim; // kv cache layer pos offset for convenience
    state.k = state.key_cache.subarray(
      kv_cache_layer_pos_offset,
      kv_cache_layer_pos_offset + kv_dim,
    );
    state.v = state.value_cache.subarray(
      kv_cache_layer_pos_offset,
      kv_cache_layer_pos_offset + kv_dim,
    );

    // qkv matmuls for this position
    //console.log("QKV matmuls");
    quantize(state.x_q, state.xb, conf.hidden_size, conf.group_size);
    qmatmul(
      state.q,
      state.x_q,
      weights.q_proj[l],
      conf.hidden_size,
      conf.hidden_size,
      conf.group_size,
    );
    qmatmul(
      state.k,
      state.x_q,
      weights.k_proj[l],
      conf.hidden_size,
      kv_dim,
      conf.group_size,
    );
    qmatmul(
      state.v,
      state.x_q,
      weights.v_proj[l],
      conf.hidden_size,
      kv_dim,
      conf.group_size,
    );

    // LoRA adapter
    if (adapter_config && adapter_state && adapter_weights) {
      //console.log("LoRA matmuls");
      if (adapter_weights.q_proj_a && adapter_weights.q_proj_b) {
        lora(
          state.q,
          state.xb,
          adapter_state.q_r,
          adapter_weights.q_proj_a[l],
          adapter_weights.q_proj_b[l],
          conf.hidden_size,
          conf.hidden_size,
          adapter_config.rank,
          adapter_scale,
        );
      }
      if (adapter_weights.k_proj_a && adapter_weights.k_proj_b) {
        lora(
          state.k,
          state.xb,
          adapter_state.k_r,
          adapter_weights.k_proj_a[l],
          adapter_weights.k_proj_b[l],
          conf.hidden_size,
          kv_dim,
          adapter_config.rank,
          adapter_scale,
        );
      }
      if (adapter_weights.v_proj_a && adapter_weights.v_proj_b) {
        lora(
          state.v,
          state.xb,
          adapter_state.v_r,
          adapter_weights.v_proj_a[l],
          adapter_weights.v_proj_b[l],
          conf.hidden_size,
          kv_dim,
          adapter_config.rank,
          adapter_scale,
        );
      }
    }

    rope(
      state.q,
      state.k,
      pos,
      conf.n_heads,
      conf.n_kv_heads,
      head_size,
      conf.rope_theta,
    );

    mutlihead_attention(
      state.xb,
      state.q,
      state.key_cache.subarray(
        kv_cache_layer_offset,
        kv_cache_layer_offset + conf.seq_len * kv_dim,
      ),
      state.value_cache.subarray(
        kv_cache_layer_offset,
        kv_cache_layer_offset + conf.seq_len * kv_dim,
      ),
      state.att,
      pos,
      conf.seq_len,
      conf.hidden_size,
      conf.n_heads,
      conf.n_kv_heads,
    );

    // final matmul to get the output of the attention
    //console.log("Attention output");
    quantize(state.x_q, state.xb, conf.hidden_size, conf.group_size);
    qmatmul(
      state.xb2,
      state.x_q,
      weights.o_proj[l],
      conf.hidden_size,
      conf.hidden_size,
      conf.group_size,
    );

    if (adapter_config && adapter_state && adapter_weights) {
      if (adapter_weights.o_proj_a && adapter_weights.o_proj_b) {
        lora(
          state.xb2,
          state.xb,
          adapter_state.o_r,
          adapter_weights.o_proj_a[l],
          adapter_weights.o_proj_b[l],
          conf.hidden_size,
          conf.hidden_size,
          adapter_config.rank,
          adapter_scale,
        );
      }
    }

    //console.log("Residual connections");
    // gemma2: post_attention_layernorm is applied before the residual connection,
    accum(state.x, state.xb2, conf.hidden_size);

    //console.log("FFN rmsnorm");
    // gemma2: pre_feedforward_layernorm is applied after the residual connection
    rmsnorm(
      state.xb,
      state.x,
      weights.post_attention_layernorm[l],
      conf.hidden_size,
      conf.norm_eps,
    );

    //console.log("FFN Gate + Up");
    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    quantize(state.x_q, state.xb, conf.hidden_size, conf.group_size);
    qmatmul(
      state.hb,
      state.x_q,
      weights.gate_proj[l],
      conf.hidden_size,
      hidden_dim,
      conf.group_size,
    );
    qmatmul(
      state.hb2,
      state.x_q,
      weights.up_proj[l],
      conf.hidden_size,
      hidden_dim,
      conf.group_size,
    );

    if (adapter_config && adapter_state && adapter_weights) {
      if (adapter_weights.gate_proj_a && adapter_weights.gate_proj_b) {
        lora(
          state.hb,
          state.xb,
          adapter_state.gate_r,
          adapter_weights.gate_proj_a[l],
          adapter_weights.gate_proj_b[l],
          conf.hidden_size,
          hidden_dim,
          adapter_config.rank,
          adapter_scale,
        );
      }
      if (adapter_weights.up_proj_a && adapter_weights.up_proj_b) {
        lora(
          state.hb2,
          state.xb,
          adapter_state.up_r,
          adapter_weights.up_proj_a[l],
          adapter_weights.up_proj_b[l],
          conf.hidden_size,
          hidden_dim,
          adapter_config.rank,
          adapter_scale,
        );
      }
    }

    if (conf.hidden_act === "silu") {
      silu(state.hb, hidden_dim);
    } else if (
      conf.hidden_act === "gelu" || conf.hidden_act === "gelu_pytorch_tanh"
    ) {
      gelu(state.hb, hidden_dim);
    } else {
      throw new Error("Unknown hidden activation function");
    }

    // elementwise multiply with w3(x)
    elemmul(state.hb, state.hb2, hidden_dim);

    //console.log("FFN Final output");
    // final matmul to get the output of the ffn
    quantize(state.hb_q, state.hb, hidden_dim, conf.group_size);
    qmatmul(
      state.xb,
      state.hb_q,
      weights.down_proj[l],
      hidden_dim,
      conf.hidden_size,
      conf.group_size,
    );

    if (adapter_config && adapter_state && adapter_weights) {
      if (adapter_weights.down_proj_a && adapter_weights.down_proj_b) {
        lora(
          state.xb,
          state.hb,
          adapter_state.down_r,
          adapter_weights.down_proj_a[l],
          adapter_weights.down_proj_b[l],
          hidden_dim,
          conf.hidden_size,
          adapter_config.rank,
          adapter_scale,
        );
      }
    }
    // gemma2: post_feedforward_layernorm here before the residual connection
    //console.log("Residual connections");
    accum(state.x, state.xb, conf.hidden_size);
  }

  // final rmsnorm
  rmsnorm(state.x, state.x, weights.norm, conf.hidden_size, conf.norm_eps);

  // classifier into logits
  if (weights.lm_head instanceof WasmQ8Tensor) {
    quantize(state.x_q, state.x, conf.hidden_size, conf.group_size);
    qmatmul(
      state.logits,
      state.x_q,
      weights.lm_head,
      conf.hidden_size,
      conf.vocab_size,
      conf.group_size,
    );
  } else {
    matmul(
      state.logits,
      state.x,
      weights.lm_head,
      conf.hidden_size,
      conf.vocab_size,
    );
  }
  // gemma2: self.config.final_logit_softcapping
}

async function main() {
  //console.log(Deno.args);
  const [checkpoint, ...args] = Deno.args;
  let temperature = 1.0; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  let topp = 1.0; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  let rng_seed = 0; // seed rng with time by default
  let steps = 256; // max number of steps to run for, 0: use seq_len
  let prompt: string | null = null; // prompt string
  let adapter = null;
  let direct_read = false;

  if (!checkpoint) return error_usage();
  for (let i = 0; i < args.length; i += 2) {
    if (i + 1 >= args.length) return error_usage(); // must have arg after flag
    let [arg, val] = [args[i], args[i + 1]];
    if (arg.charAt(0) != "-") return error_usage(); // must start with dash
    if (arg.length != 2) return error_usage(); // must be -x (one dash, one letter)
    // read in the args
    switch (args[i][1]) {
      case "t":
        temperature = parseFloat(val);
        break;
      case "p":
        topp = parseFloat(val);
        break;
      case "s":
        rng_seed = parseInt(val);
        break;
      case "n":
        steps = parseInt(val);
        break;
      case "i":
        prompt = val;
        break;
      case "a":
        adapter = val;
        break;
      case "d":
        direct_read = true;
        break;
      default:
        return error_usage();
    }
  }
  set_seed(rng_seed || Date.now());

  console.log('Loading model from "%s"...', checkpoint);
  const { config, weights } = await readModel(checkpoint, direct_read);

  let adapterConfig;
  let adapterWeights;
  if (adapter) {
    console.log('Loading adapter from "%s"...', adapter);
    //const adptr = readAdapter(adapter, config); // TODO this needs to be a hf config, but we don't have that here anymore
    //adapterConfig = adptr.adapterConfig;
    //adapterWeights = adptr.adapterWeights;
  }

  // read in the tokenizer.bin file
  console.log("Loading tokenizer from tokenizer.bin...");
  const hfTokenizer = readHFRepoTokenizer(
    checkpoint + "/tokenizer_config.json",
    checkpoint + "/tokenizer.json",
  );

  // right now we cannot run for more than config.seq_len steps
  if (steps <= 0 || steps > config.seq_len) steps = config.seq_len;

  // create and init the application RunState
  const state = newRunState(config);
  const adapterState = adapterConfig
    ? newAdapterState(adapterConfig)
    : undefined;
  if (prompt == null) prompt = "";

  const prompt_tokens = encode(
    prompt,
    hfTokenizer,
  );
  console.log("Prompt tokens:", prompt_tokens);
  //console.log(prompt_tokens, num_prompt_tokens);

  // start the main loop
  console.log("Starting inference:");
  let start = 0; // used to time our code, only initialized after first iteration
  let next; // will store the next token in the sequence
  let token = prompt_tokens[0]; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
  let pos = 0; // position in the sequence
  while (pos < steps) {
    //console.log("Step %d", pos + 1);
    // forward the transformer to get logits for the next token
    forward(
      token,
      pos,
      config,
      state,
      weights,
      adapterConfig,
      adapterState,
      adapterWeights,
    );

    //console.log("Step %d decoding", pos + 1);

    // advance the state machine
    if (pos < prompt_tokens.length - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // sample the next token
      next = sample(
        state.logits,
        config.vocab_size,
        temperature,
        topp,
      );
    }
    pos++;

    // data-dependent terminating condition: the BOS (1) or EOS (2) token delimits sequences
    //if (next == 1) break;
    //if (next == 2) break;

    // print the token as string, decode it with the Tokenizer object
    const piece = decode(token, next, hfTokenizer);

    writeAllSync(Deno.stdout, new TextEncoder().encode(piece)); // note: assumes utf8 terminal
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) start = Date.now();
  }

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  console.log(
    "\n\nachieved tok/s: %f\n",
    (pos - 1) / (Date.now() - start) * 1000.0,
  );
}

function error_usage(): never {
  console.error("Usage: ... llama2.ts <checkpoint> [options]");
  console.error('Example: llama2.ts ./model/ -n 256 -i "Once upon a time"');
  console.error("Options:");
  console.error("  -a <string> adapter checkpoint (optional, default: none)");
  console.error("  -t <float>  temperature, default 1.0");
  console.error(
    "  -p <float>  p value in top-p (nucleus) sampling. default 0.9, 0 = off",
  );
  console.error("  -s <int>    random seed, default time(NULL)");
  console.error(
    "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len",
  );
  console.error("  -i <string> input prompt");
  Deno.exit(1);
}

main();
