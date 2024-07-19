// Llama2 transformer model inference in one TypeScript file.
// by Luca Elin Haneklau, 2024 (MIT licensed).
// Based on Oleksandr Nikitin's llama2.ts: https://github.com/wizzard0/llama2.ts
// Based on Andrej Karpathy's llama2.c: https://github.com/karpathy/llama2.c

// deno-lint-ignore-file prefer-const

import * as fs from "node:fs";
import { writeAllSync } from "https://deno.land/std/io/mod.ts";
import {
  permuteReverse,
  readHFRepo,
  TransformersModel,
} from "./safetensors.ts";
import { accum, elemmul, matmul, qmatmul, rmsnorm, silu } from "./kernels.ts";
import { lora, mutlihead_attention, rope } from "./transformer.ts";
import { decode, encode } from "./sentencepiece.ts";
import { sample } from "./sampler.ts";
import { set_seed } from "./rng.ts";
import {
  newQ8Array,
  newQ8ArrayFrom,
  Q8Array,
  quantize,
} from "./quantization.ts";

// ----------------------------------------------------------------------------
// binary utils

type float = number;
type int = number;

interface Config {
  dim: int;
  hidden_dim: int;
  n_layers: int;
  n_heads: int;
  n_kv_heads: int;
  vocab_size: int;
  seq_len: int;
  head_size: int;
  rope_theta: float;
  group_size: int;
  norm_eps?: float;
}

interface TransformerWeights {
  embed_tokens: Float32Array;
  input_layernorm: Float32Array[];
  q_proj: Float32Array[];
  k_proj: Float32Array[];
  v_proj: Float32Array[];
  o_proj: Float32Array[];
  post_attention_layernorm: Float32Array[];
  gate_proj: Float32Array[];
  down_proj: Float32Array[];
  up_proj: Float32Array[];
  norm: Float32Array;
  lm_head: Float32Array;
}
interface QuantizedTransformerWeights {
  embed_tokens: Float32Array;
  input_layernorm: Float32Array[];
  q_proj: Q8Array[];
  k_proj: Q8Array[];
  v_proj: Q8Array[];
  o_proj: Q8Array[];
  post_attention_layernorm: Float32Array[];
  gate_proj: Q8Array[];
  down_proj: Q8Array[];
  up_proj: Q8Array[];
  norm: Float32Array;
  lm_head: Q8Array;
}

interface AdapterConfig {
  rank: int;
  alpha: int;
}

interface AdapterWeights {
  q_proj_a?: Float32Array[];
  q_proj_b?: Float32Array[];
  k_proj_a?: Float32Array[];
  k_proj_b?: Float32Array[];
  v_proj_a?: Float32Array[];
  v_proj_b?: Float32Array[];
  down_proj_a?: Float32Array[];
  down_proj_b?: Float32Array[];
  o_proj_a?: Float32Array[];
  o_proj_b?: Float32Array[];
  gate_proj_a?: Float32Array[];
  gate_proj_b?: Float32Array[];
  up_proj_a?: Float32Array[];
  up_proj_b?: Float32Array[];
}

interface RunState {
  x: Float32Array;
  x_q: Q8Array;
  xb: Float32Array;
  xb2: Float32Array;
  hb: Float32Array;
  hb_q: Q8Array;
  hb2: Float32Array;
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  att: Float32Array;
  logits: Float32Array;
  key_cache: Float32Array;
  value_cache: Float32Array;
}

interface AdapterRunState {
  q_r: Float32Array;
  k_r: Float32Array;
  v_r: Float32Array;
  o_r: Float32Array;
  gate_r: Float32Array;
  down_r: Float32Array;
  up_r: Float32Array;
}

function newRunState(config: Config): RunState {
  const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
  const s = {} as RunState;
  s.x = new Float32Array(config.dim);
  s.x_q = newQ8Array(config.dim, config.group_size);
  s.xb = new Float32Array(config.dim);
  s.xb2 = new Float32Array(config.dim);
  s.hb = new Float32Array(config.hidden_dim);
  s.hb_q = newQ8Array(config.hidden_dim, config.group_size);
  s.hb2 = new Float32Array(config.hidden_dim);
  s.q = new Float32Array(config.dim);
  s.k = new Float32Array(kv_dim);
  s.v = new Float32Array(kv_dim);
  s.att = new Float32Array(config.n_heads * config.seq_len);
  s.logits = new Float32Array(config.vocab_size);
  s.key_cache = new Float32Array(config.n_layers * config.seq_len * kv_dim);
  s.value_cache = new Float32Array(
    config.n_layers * config.seq_len * kv_dim,
  );
  return s;
}
function newAdapterState(config: AdapterConfig): AdapterRunState {
  const s = {} as AdapterRunState;
  s.q_r = new Float32Array(config.rank);
  s.k_r = new Float32Array(config.rank);
  s.v_r = new Float32Array(config.rank);
  s.down_r = new Float32Array(config.rank);
  s.o_r = new Float32Array(config.rank);
  s.gate_r = new Float32Array(config.rank);
  s.up_r = new Float32Array(config.rank);
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
  const kv_dim = (conf.dim * conf.n_kv_heads) / conf.n_heads;

  const hidden_dim = conf.hidden_dim;
  const head_size = conf.dim / conf.n_heads;

  const adapter_scale = adapter_config
    ? adapter_config.rank / adapter_config.alpha
    : 0;

  // copy the token embedding into x
  state.x.set(
    weights.embed_tokens.subarray(
      token * conf.dim,
      token * conf.dim + conf.dim,
    ),
  );

  //debugger;
  // forward all the layers
  for (let l = 0; l < conf.n_layers; l++) {
    //console.log("Layer %d", l + 1);
    // attention rmsnorm
    rmsnorm(state.xb, state.x, weights.input_layernorm[l], conf.dim);

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
    quantize(state.x_q, state.xb, conf.dim, conf.group_size);
    qmatmul(
      state.q,
      state.x_q,
      weights.q_proj[l],
      conf.dim,
      conf.dim,
      conf.group_size,
    );
    qmatmul(
      state.k,
      state.x_q,
      weights.k_proj[l],
      conf.dim,
      kv_dim,
      conf.group_size,
    );
    qmatmul(
      state.v,
      state.x_q,
      weights.v_proj[l],
      conf.dim,
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
          conf.dim,
          conf.dim,
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
          conf.dim,
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
          conf.dim,
          kv_dim,
          adapter_config.rank,
          adapter_scale,
        );
      }
    }

    rope(state.q, state.k, pos, conf.dim, kv_dim, head_size, conf.rope_theta);

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
      conf.dim,
      conf.n_heads,
      conf.n_kv_heads,
    );

    // final matmul to get the output of the attention
    //console.log("Attention output");
    quantize(state.x_q, state.xb, conf.dim, conf.group_size);
    qmatmul(
      state.xb2,
      state.x_q,
      weights.o_proj[l],
      conf.dim,
      conf.dim,
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
          conf.dim,
          conf.dim,
          adapter_config.rank,
          adapter_scale,
        );
      }
    }

    //console.log("Residual connections");
    accum(state.x, state.xb2, conf.dim);

    //console.log("FFN rmsnorm");
    rmsnorm(state.xb, state.x, weights.post_attention_layernorm[l], conf.dim);

    //console.log("FFN Gate + Up");
    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    quantize(state.x_q, state.xb, conf.dim, conf.group_size);
    qmatmul(
      state.hb,
      state.x_q,
      weights.gate_proj[l],
      conf.dim,
      hidden_dim,
      conf.group_size,
    );
    qmatmul(
      state.hb2,
      state.x_q,
      weights.up_proj[l],
      conf.dim,
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
          conf.dim,
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
          conf.dim,
          hidden_dim,
          adapter_config.rank,
          adapter_scale,
        );
      }
    }

    silu(state.hb, hidden_dim);

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
      conf.dim,
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
          conf.dim,
          adapter_config.rank,
          adapter_scale,
        );
      }
    }

    //console.log("Residual connections");
    accum(state.x, state.xb, conf.dim);
  }

  // final rmsnorm
  rmsnorm(state.x, state.x, weights.norm, conf.dim);

  // classifier into logits
  quantize(state.x_q, state.x, conf.dim, conf.group_size);
  qmatmul(
    state.logits,
    state.x_q,
    weights.lm_head,
    conf.dim,
    conf.vocab_size,
    conf.group_size,
  );
}

function readTokenizer(tokenizerPath: string, vocab_size: number) {
  let vocab = new Array<string>(vocab_size);
  let vocab_scores = new Array<number>(vocab_size);
  let tokBuffer = fs.readFileSync(tokenizerPath);
  let tokBufferOffset = 0;
  let _ignored_max_token_length = tokBuffer.readInt32LE(tokBufferOffset);
  tokBufferOffset += 4;
  for (let i = 0; i < Math.min(vocab_size, 32000); i++) { // TODO vocab_size is limited to 32000
    vocab_scores[i] = tokBuffer.readFloatLE(tokBufferOffset);
    tokBufferOffset += 4;

    const token_length = tokBuffer.readInt32LE(tokBufferOffset);
    tokBufferOffset += 4;

    const bytes = new Uint8Array(token_length);
    for (let j = 0; j < token_length; j++) {
      bytes[j] = tokBuffer.readUInt8(tokBufferOffset);
      tokBufferOffset += 1;
    }

    vocab[i] = new TextDecoder().decode(bytes);
  }

  return { vocab, vocab_scores };
}

function readModel(
  hfConfig: TransformersModel<"F32">["config"],
  hfWeights: TransformersModel<"F32">["weights"],
) {
  const config: Config = {
    dim: hfConfig.hidden_size as number,
    hidden_dim: hfConfig.intermediate_size as number,
    head_size: hfConfig.hidden_size as number /
      (hfConfig.num_attention_heads as number),
    n_heads: hfConfig.num_attention_heads as number,
    n_kv_heads: hfConfig.num_key_value_heads as number,
    n_layers: hfConfig.num_hidden_layers as number,
    seq_len: hfConfig.max_position_embeddings as number,
    vocab_size: 32000, // hfConfig.vocab_size as number,
    rope_theta: hfConfig.rope_theta as number,
    group_size: hfConfig.group_size as number ?? 32, // Andrej uses 64, according to thebloke 128 is optimal, the current implementation degrades at 32 and above
    norm_eps: hfConfig.rms_norm_eps as number,
  };

  const weights: TransformerWeights = {
    embed_tokens: hfWeights.model.embed_tokens.weight.weights,
    input_layernorm: hfWeights.model.layers.map((l) =>
      l.input_layernorm.weight.weights
    ),
    q_proj: hfWeights.model.layers.map((l) =>
      permuteReverse(
        l.self_attn.q_proj?.weight.weights ??
          l.self_attn.qkv_proj.weight.weights.slice(
            0,
            config.dim * config.dim,
          ),
        config.n_heads,
        config.dim,
        config.dim,
      )
    ),
    k_proj: hfWeights.model.layers.map((l) =>
      permuteReverse(
        l.self_attn.k_proj?.weight.weights ??
          l.self_attn.qkv_proj.weight.weights.slice(
            config.dim * config.dim,
            config.dim * config.dim +
              config.dim *
                (Math.floor(config.dim / config.n_heads) * config.n_kv_heads),
          ),
        config.n_kv_heads,
        Math.floor(config.dim / config.n_heads) * config.n_kv_heads,
        config.dim,
      )
    ),
    v_proj: hfWeights.model.layers.map((l) =>
      l.self_attn.v_proj?.weight.weights ??
        l.self_attn.qkv_proj.weight.weights.slice(
          config.dim * config.dim +
            config.dim *
              (Math.floor(config.dim / config.n_heads) * config.n_kv_heads),
          config.dim * config.dim +
            config.dim *
              (Math.floor(config.dim / config.n_heads) * config.n_kv_heads) * 2,
        )
    ),
    o_proj: hfWeights.model.layers.map((l) =>
      l.self_attn.o_proj.weight.weights
    ),
    post_attention_layernorm: hfWeights.model.layers.map((l) =>
      l.post_attention_layernorm.weight.weights
    ),
    gate_proj: hfWeights.model.layers.map((l) =>
      l.mlp.gate_proj?.weight.weights ??
        l.mlp.gate_up_proj?.weight?.weights.slice(
          0,
          config.hidden_dim * config.dim,
        )
    ),
    down_proj: hfWeights.model.layers.map((l) =>
      l.mlp.down_proj.weight.weights
    ),
    up_proj: hfWeights.model.layers.map((l) =>
      l.mlp.up_proj?.weight?.weights ??
        l.mlp.gate_up_proj?.weight.weights.slice(
          config.hidden_dim * config.dim,
          2 * config.hidden_dim * config.dim,
        )
    ),
    norm: hfWeights.model.norm.weight.weights,
    lm_head: hfWeights.lm_head?.weight?.weights ??
      hfWeights.model.embed_tokens.weight.weights,
  };

  return { config, weights };
}

function readModelQuantized(
  hfConfig: TransformersModel<"F32">["config"],
  hfWeights: TransformersModel<"F32">["weights"],
  hfWeightsQuant: TransformersModel<"Q8_0">["weights"],
  hfMetadata: TransformersModel<"Q8_0">["metadata"],
) {
  const config: Config = {
    dim: hfConfig.hidden_size as number,
    hidden_dim: hfConfig.intermediate_size as number,
    head_size: hfConfig.hidden_size as number /
      (hfConfig.num_attention_heads as number),
    n_heads: hfConfig.num_attention_heads as number,
    n_kv_heads: hfConfig.num_key_value_heads as number,
    n_layers: hfConfig.num_hidden_layers as number,
    seq_len: hfConfig.max_position_embeddings as number,
    vocab_size: 32000, // hfConfig.vocab_size as number,
    rope_theta: hfConfig.rope_theta as number,
    group_size: parseInt(hfMetadata.group_size),
    norm_eps: hfConfig.rms_norm_eps as number,
  };

  const weights: QuantizedTransformerWeights = {
    embed_tokens: hfWeights.model.embed_tokens.weight.weights,
    input_layernorm: hfWeights.model.layers.map((l) =>
      l.input_layernorm.weight.weights
    ),
    q_proj: hfWeightsQuant.model.layers.map((l) =>
      permuteReverse(
        l.self_attn.q_proj?.weight.weights ??
          l.self_attn.qkv_proj.weight.weights.slice(
            0,
            config.dim * config.dim,
          ),
        config.n_heads,
        config.dim,
        config.dim,
      )
    ),
    k_proj: hfWeightsQuant.model.layers.map((l) =>
      permuteReverse(
        l.self_attn.k_proj?.weight.weights ??
          l.self_attn.qkv_proj.weight.weights.slice(
            config.dim * config.dim,
            config.dim * config.dim +
              config.dim *
                (Math.floor(config.dim / config.n_heads) * config.n_kv_heads),
          ),
        config.n_kv_heads,
        Math.floor(config.dim / config.n_heads) * config.n_kv_heads,
        config.dim,
      )
    ),
    v_proj: hfWeightsQuant.model.layers.map(
      (l) =>
        l.self_attn.v_proj?.weight.weights ??
          l.self_attn.qkv_proj.weight.weights.slice(
            config.dim * config.dim +
              config.dim *
                (Math.floor(config.dim / config.n_heads) * config.n_kv_heads),
            config.dim * config.dim +
              config.dim *
                (Math.floor(config.dim / config.n_heads) * config.n_kv_heads) *
                2,
          ),
    ),
    o_proj: hfWeightsQuant.model.layers.map(
      (l) => l.self_attn.o_proj.weight.weights,
    ),
    post_attention_layernorm: hfWeights.model.layers.map((l) =>
      l.post_attention_layernorm.weight.weights
    ),
    gate_proj: hfWeightsQuant.model.layers.map(
      (l) =>
        l.mlp.gate_proj?.weight.weights ??
          l.mlp.gate_up_proj?.weight?.weights.slice(
            0,
            config.hidden_dim * config.dim,
          ),
    ),
    down_proj: hfWeightsQuant.model.layers.map(
      (l) => l.mlp.down_proj.weight.weights,
    ),
    up_proj: hfWeightsQuant.model.layers.map(
      (l) =>
        l.mlp.up_proj?.weight?.weights ??
          l.mlp.gate_up_proj?.weight.weights.slice(
            config.hidden_dim * config.dim,
            2 * config.hidden_dim * config.dim,
          ),
    ),
    norm: hfWeights.model.norm.weight.weights,
    lm_head: hfWeightsQuant.lm_head?.weight?.weights ??
      hfWeightsQuant.model.embed_tokens.weight.weights,
  };
  return { config, weights };
}

function readAdapter(
  hfConfig: TransformersModel<"F32">["config"],
  hfAdapterConfig: TransformersModel<"F32">["config"],
  hfAdapterWeights: TransformersModel<"F32">["weights"],
) {
  const adapterConfig: AdapterConfig = {
    rank: hfAdapterConfig.r as number,
    alpha: hfAdapterConfig.lora_alpha as number,
  };

  const adapterWeights: AdapterWeights = {
    q_proj_a: (hfAdapterConfig.target_modules as string[]).includes("q_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        permuteReverse(
          new Float32Array(l.self_attn.q_proj.lora_A.weight.weights),
          hfConfig.num_attention_heads as number,
          hfConfig.hidden_size as number,
          hfAdapterConfig.r as number,
        )
      )
      : undefined,
    q_proj_b: (hfAdapterConfig.target_modules as string[]).includes("q_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        permuteReverse(
          new Float32Array(l.self_attn.q_proj.lora_B.weight.weights),
          hfConfig.num_attention_heads as number,
          hfConfig.hidden_size as number,
          hfAdapterConfig.r as number,
        )
      )
      : undefined,
    k_proj_a: (hfAdapterConfig.target_modules as string[]).includes("k_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        permuteReverse(
          new Float32Array(l.self_attn.k_proj.lora_A.weight.weights),
          hfConfig.num_attention_heads as number,
          hfConfig.hidden_size as number,
          hfAdapterConfig.r as number,
        )
      )
      : undefined,
    k_proj_b: (hfAdapterConfig.target_modules as string[]).includes("k_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        permuteReverse(
          new Float32Array(l.self_attn.k_proj.lora_B.weight.weights),
          hfConfig.num_attention_heads as number,
          hfConfig.hidden_size as number,
          hfAdapterConfig.r as number,
        )
      )
      : undefined,
    v_proj_a: (hfAdapterConfig.target_modules as string[]).includes("v_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        new Float32Array(l.self_attn.v_proj.lora_A.weight.weights)
      )
      : undefined,
    v_proj_b: (hfAdapterConfig.target_modules as string[]).includes("v_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        new Float32Array(l.self_attn.v_proj.lora_B.weight.weights)
      )
      : undefined,
    o_proj_a: (hfAdapterConfig.target_modules as string[]).includes("o_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        new Float32Array(l.self_attn.o_proj.lora_A.weight.weights)
      )
      : undefined,
    o_proj_b: (hfAdapterConfig.target_modules as string[]).includes("o_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        new Float32Array(l.self_attn.o_proj.lora_B.weight.weights)
      )
      : undefined,
    gate_proj_a:
      (hfAdapterConfig.target_modules as string[]).includes("gate_proj")
        ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
          new Float32Array(l.mlp.gate_proj.lora_A.weight.weights)
        )
        : undefined,
    gate_proj_b:
      (hfAdapterConfig.target_modules as string[]).includes("gate_proj")
        ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
          new Float32Array(l.mlp.gate_proj.lora_B.weight.weights)
        )
        : undefined,
    down_proj_a:
      (hfAdapterConfig.target_modules as string[]).includes("down_proj")
        ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
          new Float32Array(l.mlp.down_proj.lora_A.weight.weights)
        )
        : undefined,
    down_proj_b:
      (hfAdapterConfig.target_modules as string[]).includes("down_proj")
        ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
          new Float32Array(l.mlp.down_proj.lora_B.weight.weights)
        )
        : undefined,
    up_proj_a: (hfAdapterConfig.target_modules as string[]).includes("up_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        new Float32Array(l.mlp.up_proj.lora_A.weight.weights)
      )
      : undefined,
    up_proj_b: (hfAdapterConfig.target_modules as string[]).includes("up_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        new Float32Array(l.mlp.up_proj.lora_B.weight.weights)
      )
      : undefined,
  };

  return { adapterConfig, adapterWeights };
}

function main() {
  //console.log(Deno.args);
  const [checkpoint, ...args] = Deno.args;
  let temperature = 1.0; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  let topp = 1.0; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  let rng_seed = 0; // seed rng with time by default
  let steps = 256; // max number of steps to run for, 0: use seq_len
  let prompt: string | null = null; // prompt string
  let adapter = null;

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
      default:
        return error_usage();
    }
  }
  set_seed(rng_seed || Date.now());

  console.log('Loading model from "%s"...', checkpoint);
  const { config: hfConfig, metadata: hfMetadata, weights: hfWeightsQuant } =
    readHFRepo(
      checkpoint + "/config.json",
      checkpoint + "/model.safetensors",
      "Q8_0",
      {
        filter: (layer) =>
          layer.includes("_proj.") || layer.includes("lm_head.") ||
          layer.includes("embed_tokens."),
      },
    );
  const { weights: hfWeights } = readHFRepo(
    checkpoint + "/config.json",
    checkpoint + "/model.safetensors",
    "F32",
    {
      /*filter: (layer) =>
        !(layer.includes("_proj.") || layer.includes("lm_head.")),*/
    },
  );
  const { config, weights } = readModelQuantized(
    hfConfig,
    hfWeights,
    hfWeightsQuant,
    hfMetadata,
  );

  let adapterConfig;
  let adapterWeights;
  if (adapter) {
    console.log('Loading adapter from "%s"...', adapter);
    const { config: hfAdapterConfig, weights: hfAdapterWeights } = readHFRepo(
      adapter + "/adapter_config.json",
      adapter + "/adapter_model.safetensors",
      "F32",
    );
    const model = readAdapter(
      hfConfig,
      hfAdapterConfig,
      hfAdapterWeights,
    );
    adapterConfig = model.adapterConfig;
    adapterWeights = model.adapterWeights;
  }

  // read in the tokenizer.bin file
  console.log("Loading tokenizer from tokenizer.bin...");
  const { vocab, vocab_scores } = readTokenizer(
    "tokenizer.bin",
    config.vocab_size,
  );

  // right now we cannot run for more than config.seq_len steps
  if (steps <= 0 || steps > config.seq_len) steps = config.seq_len;

  // create and init the application RunState
  const state = newRunState(config);
  const adapterState = adapterConfig
    ? newAdapterState(adapterConfig)
    : undefined;
  if (prompt == null) prompt = "";

  // encode the (string) prompt into tokens sequence
  let num_prompt_tokens = 0;
  let prompt_tokens: Int32Array = new Int32Array(prompt.length + 3); // +3 for '\0', ?BOS, ?EOS

  num_prompt_tokens = encode(
    prompt,
    true, // bos
    false, // no eos
    vocab,
    vocab_scores,
    prompt_tokens,
  );
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
    if (pos < num_prompt_tokens - 1) {
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

    // data-dependent terminating condition: the BOS (1) token delimits sequences
    if (next == 1) break;

    // print the token as string, decode it with the Tokenizer object
    const piece = decode(vocab, token, next);

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
