// deno-lint-ignore-file no-explicit-any
import { config } from "node:process";
import {
  dequantize,
  readHFRepo,
  SUPPORTED_DTYPES,
  TransformersModel,
  writeSafetensors,
} from "./safetensors.ts";
import { JSF32Tensor, JSQ8Tensor } from "./types.ts";
import { WasmF32Tensor, WasmQ8Tensor } from "./types_wasm.ts";

type F32Tensor = WasmF32Tensor;
type Q8Tensor = WasmQ8Tensor;

// ----------------------------------------------------------------------------
// binary utils

type float = number;
type int = number;

function raise(message: string): never {
  throw new Error(message);
}

export interface Config {
  hidden_size: int;
  intermediate_size: int;
  n_layers: int;
  n_heads: int;
  n_kv_heads: int;
  vocab_size: int;
  seq_len: int;
  head_size: int;
  rope_theta: float;
  group_size: int;
  norm_eps: float;
  hidden_act: "silu" | "gelu" | "gelu_pytorch_tanh";
  model_type: "llama" | "gemma";
}

export interface TransformerWeights {
  embed_tokens: F32Tensor;
  input_layernorm: F32Tensor[];
  q_proj: F32Tensor[];
  k_proj: F32Tensor[];
  v_proj: F32Tensor[];
  o_proj: F32Tensor[];
  post_attention_layernorm: F32Tensor[];
  gate_proj: F32Tensor[];
  down_proj: F32Tensor[];
  up_proj: F32Tensor[];
  norm: F32Tensor;
  lm_head: F32Tensor;
}
export interface QuantizedTransformerWeights {
  embed_tokens: F32Tensor | Q8Tensor;
  input_layernorm: F32Tensor[];
  q_proj: Q8Tensor[];
  k_proj: Q8Tensor[];
  v_proj: Q8Tensor[];
  o_proj: Q8Tensor[];
  post_attention_layernorm: F32Tensor[];
  gate_proj: Q8Tensor[];
  down_proj: Q8Tensor[];
  up_proj: Q8Tensor[];
  norm: F32Tensor;
  lm_head: F32Tensor | Q8Tensor;
}

export async function readModel(checkpoint: string, direct_read = false) {
  if (!direct_read) {
    try {
      const { config: hfConfig, metadata: hfMetadata, weights: hfWeights } =
        readHFRepo(
          checkpoint + "/.converted/config.json",
          checkpoint + "/.converted/model.safetensors",
        );

      if (hfConfig["llama2.ts"]) {
        hfConfig.model_type = hfConfig.model_type + "_fixed";
        const weights = Object.fromEntries(
          Object.entries(hfWeights.model).map(([name, value]) => {
            if ("0" in value) {
              if (value[0].weight.weights instanceof Float32Array) {
                return [
                  name,
                  Object.values(value).map((t) =>
                    loadF32Wasm(t.weight.weights)
                  ),
                ];
              } else {
                return [
                  name,
                  Object.values(value).map((t) => loadQ8Wasm(t.weight.weights)),
                ];
              }
            } else {
              if (value.weight.weights instanceof Float32Array) {
                return [name, loadF32Wasm(value.weight.weights)];
              } else {
                return [name, loadQ8Wasm(value.weight.weights)];
              }
            }
          }),
        );
        hfConfig.model_type = (hfConfig.model_type as string).replace(
          "_fixed",
          "",
        );
        return { weights, config: hfConfig };
      }
    } catch (e) {
      console.log(
        "Failed to load from converted model cache.",
      );
    }
  }

  console.log("Reading HF model...");
  const { config: hfConfig, metadata: hfMetadata, weights: hfWeights } =
    readHFRepo(
      checkpoint + "/config.json",
      checkpoint + "/model.safetensors",
    );

  console.log("Quantizing model...");
  const model = convertModelQuantized(
    hfConfig,
    hfWeights,
    hfMetadata,
  );

  if (!direct_read) {
    console.log("Writing quantized model cache...");
    const weightsToStore = Object.fromEntries(
      Object.entries(model.weights).map(([key, value]) => {
        if (value instanceof Array) {
          return [
            key,
            value.map((t) => ({
              weight: {
                dtype: t instanceof WasmQ8Tensor ? "Q8_0" : "F32",
                weights: t instanceof WasmQ8Tensor
                  ? { q: t.q.array, s: t.s.array }
                  : t.array,
              },
            })),
          ];
        }
        return [
          key,
          {
            weight: {
              dtype: value instanceof WasmQ8Tensor ? "Q8_0" : "F32",
              weights: value instanceof WasmQ8Tensor
                ? { q: value.q.array, s: value.s.array }
                : value.array,
            },
          },
        ];
      }),
    );

    await Deno.mkdir(checkpoint + "/.converted", { recursive: true });

    await writeSafetensors(
      {
        config: { ...model.config, "llama2.ts": true } as any,
        weights: { model: weightsToStore },
        metadata: {},
      },
      checkpoint + "/.converted/config.json",
      checkpoint + "/.converted/model.safetensors",
    );
  }

  return model;
}

function convertModelQuantized(
  hfConfig: TransformersModel["config"],
  hfWeights: TransformersModel["weights"],
  hfMetadata: TransformersModel["metadata"],
) {
  const config: Config = {
    hidden_size: hfConfig.hidden_size as number,
    intermediate_size: hfConfig.intermediate_size as number,
    head_size: hfConfig.hidden_size as number /
      (hfConfig.num_attention_heads as number),
    n_heads: hfConfig.num_attention_heads as number,
    n_kv_heads: hfConfig.num_key_value_heads as number,
    n_layers: hfConfig.num_hidden_layers as number,
    seq_len: hfConfig.max_position_embeddings as number,
    vocab_size: hfConfig.vocab_size as number,
    rope_theta: hfConfig.rope_theta as number,
    group_size: parseInt(
      hfMetadata.group_size ?? JSQ8Tensor.DEFAULT_GROUP_SIZE.toString(),
    ),
    norm_eps: hfConfig.rms_norm_eps as number,
    hidden_act: (hfConfig.hidden_activation ?? hfConfig.hidden_act) as
      | "gelu"
      | "silu"
      | "gelu_pytorch_tanh",
    model_type: hfConfig.model_type as "llama" | "gemma",
  };

  const weights: QuantizedTransformerWeights = {
    embed_tokens: loadQ8Wasm(hfWeights.model.embed_tokens.weight.weights) ??
      raise("embed_tokens not found"),

    input_layernorm: hfWeights.model.layers.map((l) =>
      loadF32Wasm(l.input_layernorm.weight.weights) ??
        raise("input_layernorm not found")
    ),
    q_proj: hfWeights.model.layers.map((l) =>
      loadQ8Wasm(
        l.self_attn.q_proj?.weight.weights ??
          split(
            l.self_attn.qkv_proj.weight.weights,
            0,
            config.hidden_size * config.hidden_size,
          ),
      ) ?? raise("q_proj not found")
    ),
    k_proj: hfWeights.model.layers.map((l) =>
      loadQ8Wasm(
        l.self_attn.k_proj?.weight.weights ??
          split(
            l.self_attn.qkv_proj.weight.weights,
            config.hidden_size * config.hidden_size,
            config.hidden_size * config.hidden_size +
              config.hidden_size *
                (Math.floor(config.hidden_size / config.n_heads) *
                  config.n_kv_heads),
          ),
      ) ?? raise("k_proj not found")
    ),
    v_proj: hfWeights.model.layers.map(
      (l) =>
        loadQ8Wasm(
          l.self_attn.v_proj?.weight.weights ??
            split(
              l.self_attn.qkv_proj.weight.weights,
              config.hidden_size * config.hidden_size +
                config.hidden_size *
                  (Math.floor(config.hidden_size / config.n_heads) *
                    config.n_kv_heads),
              config.hidden_size * config.hidden_size +
                config.hidden_size *
                  (Math.floor(config.hidden_size / config.n_heads) *
                    config.n_kv_heads) *
                  2,
            ),
        ) ?? raise("v_proj not found"),
    ),
    o_proj: hfWeights.model.layers.map(
      (l) =>
        loadQ8Wasm(l.self_attn.o_proj.weight.weights) ??
          raise("o_proj not found"),
    ),
    post_attention_layernorm: hfWeights.model.layers.map((l) =>
      loadF32Wasm(l.post_attention_layernorm.weight.weights) ??
        raise("post_attention_layernorm not found")
    ),
    gate_proj: hfWeights.model.layers.map(
      (l) =>
        loadQ8Wasm(
          l.mlp.gate_proj?.weight.weights ??
            split(
              l.mlp.gate_up_proj?.weight?.weights,
              0,
              config.intermediate_size * config.hidden_size,
            ),
        ) ?? raise("gate_proj not found"),
    ),
    down_proj: hfWeights.model.layers.map(
      (l) =>
        loadQ8Wasm(l.mlp.down_proj.weight.weights) ??
          raise("down_proj not found"),
    ),
    up_proj: hfWeights.model.layers.map(
      (l) =>
        loadQ8Wasm(
          l.mlp.up_proj?.weight?.weights ??
            split(
              l.mlp.gate_up_proj?.weight.weights,
              config.intermediate_size * config.hidden_size,
              2 * config.intermediate_size * config.hidden_size,
            ),
        ) ?? raise("up_proj not found"),
    ),
    norm: loadF32Wasm(hfWeights.model.norm.weight.weights) ??
      raise("norm not found"),
    lm_head: loadQ8Wasm(
      hfWeights.lm_head?.weight?.weights ??
        hfWeights.model.embed_tokens.weight.weights,
    ) ?? raise("lm_head not found"),
  };
  if (config.model_type === "gemma") {
    for (let i = 0; i < weights.norm.length; i++) {
      weights.norm.array[i] = weights.norm.array[i] + 1;
    }
    weights.input_layernorm.forEach((layernorm) => {
      for (let i = 0; i < layernorm.length; i++) {
        layernorm.array[i] = layernorm.array[i] + 1;
      }
    });
    weights.post_attention_layernorm.forEach((layernorm) => {
      for (let i = 0; i < layernorm.length; i++) {
        layernorm.array[i] = layernorm.array[i] + 1;
      }
    });
  }
  if (config.model_type === "llama") {
    // undo huggingface permutation
    for (const layer of weights.q_proj) {
      permuteReverse(
        layer,
        config.n_heads,
        config.hidden_size,
        config.hidden_size,
      );
    }
    for (const layer of weights.k_proj) {
      permuteReverse(
        layer,
        config.n_kv_heads,
        Math.floor(config.hidden_size / config.n_heads) * config.n_kv_heads,
        config.hidden_size,
      );
    }
  }
  return { config, weights };
}

function split(
  weights: SUPPORTED_DTYPES | undefined,
  start: number,
  end: number,
): SUPPORTED_DTYPES | undefined {
  if (!weights) return undefined;
  if ("q" in weights) {
    const gs = weights.q.length / weights.s.length;
    return {
      q: weights.q.subarray(start, end),
      s: weights.s.subarray(Math.floor(start / gs), Math.floor(end / gs)),
    };
  } else if (weights instanceof Float32Array) {
    return weights.subarray(start, end);
  } else {
    throw new Error("Unsupported dtype");
  }
}

function loadF32Wasm(weights?: SUPPORTED_DTYPES): WasmF32Tensor | undefined {
  if (!weights) return undefined;
  if (weights instanceof Float32Array) {
    const ret = WasmF32Tensor.allocate(weights.length);
    ret.array.set(weights);
    return ret;
  } else if (weights instanceof Float16Array) {
    const ret = WasmF32Tensor.allocate(weights.length);
    ret.array.set(new Float32Array(weights));
    return ret;
  } else {
    throw new Error("Unsupported dtype " + weights.constructor.name);
  }
}
function loadF32JS(weights?: SUPPORTED_DTYPES): JSF32Tensor | undefined {
  if (!weights) return undefined;
  if (weights instanceof Float32Array) {
    const ret = JSF32Tensor.allocate(weights.length);
    ret.array.set(weights);
    return ret;
  } else if (weights instanceof Float16Array) {
    const ret = JSF32Tensor.allocate(weights.length);
    ret.array.set(new Float32Array(weights));
    return ret;
  } else {
    throw new Error("Unsupported dtype " + weights.constructor.name);
  }
}

function loadQ8Wasm(weights?: SUPPORTED_DTYPES): WasmQ8Tensor | undefined {
  if (!weights) return undefined;
  if ("q" in weights) {
    const ret = WasmQ8Tensor.allocate(
      weights.q.length,
      weights.q.length / weights.s.length,
    );
    ret.q.array.set(weights.q);
    ret.s.array.set(weights.s);
    return ret;
  } else if (weights instanceof Float32Array) {
    const ret = WasmQ8Tensor.allocateFromJSF32(new JSF32Tensor(weights));
    return ret;
  } else if (weights instanceof Float16Array) {
    const ret = WasmQ8Tensor.allocateFromJSF32(
      new JSF32Tensor(
        new Float32Array(weights),
      ),
    );
    return ret;
  } else {
    throw new Error("Unsupported dtype " + weights.constructor.name);
  }
}

export interface AdapterConfig {
  rank: int;
  alpha: int;
}

export interface AdapterWeights {
  q_proj_a?: F32Tensor[];
  q_proj_b?: F32Tensor[];
  k_proj_a?: F32Tensor[];
  k_proj_b?: F32Tensor[];
  v_proj_a?: F32Tensor[];
  v_proj_b?: F32Tensor[];
  down_proj_a?: F32Tensor[];
  down_proj_b?: F32Tensor[];
  o_proj_a?: F32Tensor[];
  o_proj_b?: F32Tensor[];
  gate_proj_a?: F32Tensor[];
  gate_proj_b?: F32Tensor[];
  up_proj_a?: F32Tensor[];
  up_proj_b?: F32Tensor[];
}

export function readAdapter(
  adapter: string,
  baseModelConfig: TransformersModel["config"],
) {
  const { config: hfAdapterConfig, weights: hfAdapterWeights } = readHFRepo(
    adapter + "/adapter_config.json",
    adapter + "/adapter_model.safetensors",
  );
  const { adapterConfig, adapterWeights } = convertAdapter(
    baseModelConfig,
    hfAdapterConfig,
    hfAdapterWeights,
  );

  return { adapterConfig, adapterWeights };
}

function convertAdapter(
  hfConfig: TransformersModel["config"],
  hfAdapterConfig: TransformersModel["config"],
  hfAdapterWeights: TransformersModel["weights"],
) {
  const adapterConfig: AdapterConfig = {
    rank: hfAdapterConfig.r as number,
    alpha: hfAdapterConfig.lora_alpha as number,
  };

  const adapterWeights: AdapterWeights = {
    q_proj_a: (hfAdapterConfig.target_modules as string[]).includes("q_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        permuteReverse(
          hfConfig.model_type === "llama",
          l.self_attn.q_proj.lora_A.weight.weights,
          hfConfig.num_attention_heads as number,
          hfConfig.hidden_size as number,
          hfAdapterConfig.r as number,
        )
      )
      : undefined,
    q_proj_b: (hfAdapterConfig.target_modules as string[]).includes("q_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        permuteReverse(
          hfConfig.model_type === "llama",
          l.self_attn.q_proj.lora_B.weight.weights,
          hfConfig.num_attention_heads as number,
          hfConfig.hidden_size as number,
          hfAdapterConfig.r as number,
        )
      )
      : undefined,
    k_proj_a: (hfAdapterConfig.target_modules as string[]).includes("k_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        permuteReverse(
          hfConfig.model_type === "llama",
          l.self_attn.k_proj.lora_A.weight.weights,
          hfConfig.num_attention_heads as number,
          hfConfig.hidden_size as number,
          hfAdapterConfig.r as number,
        )
      )
      : undefined,
    k_proj_b: (hfAdapterConfig.target_modules as string[]).includes("k_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        permuteReverse(
          hfConfig.model_type === "llama",
          l.self_attn.k_proj.lora_B.weight.weights,
          hfConfig.num_attention_heads as number,
          hfConfig.hidden_size as number,
          hfAdapterConfig.r as number,
        )
      )
      : undefined,
    v_proj_a: (hfAdapterConfig.target_modules as string[]).includes("v_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        l.self_attn.v_proj.lora_A.weight.weights
      )
      : undefined,
    v_proj_b: (hfAdapterConfig.target_modules as string[]).includes("v_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        l.self_attn.v_proj.lora_B.weight.weights
      )
      : undefined,
    o_proj_a: (hfAdapterConfig.target_modules as string[]).includes("o_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        l.self_attn.o_proj.lora_A.weight.weights
      )
      : undefined,
    o_proj_b: (hfAdapterConfig.target_modules as string[]).includes("o_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        l.self_attn.o_proj.lora_B.weight.weights
      )
      : undefined,
    gate_proj_a:
      (hfAdapterConfig.target_modules as string[]).includes("gate_proj")
        ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
          l.mlp.gate_proj.lora_A.weight.weights
        )
        : undefined,
    gate_proj_b:
      (hfAdapterConfig.target_modules as string[]).includes("gate_proj")
        ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
          l.mlp.gate_proj.lora_B.weight.weights
        )
        : undefined,
    down_proj_a:
      (hfAdapterConfig.target_modules as string[]).includes("down_proj")
        ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
          l.mlp.down_proj.lora_A.weight.weights
        )
        : undefined,
    down_proj_b:
      (hfAdapterConfig.target_modules as string[]).includes("down_proj")
        ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
          l.mlp.down_proj.lora_B.weight.weights
        )
        : undefined,
    up_proj_a: (hfAdapterConfig.target_modules as string[]).includes("up_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        l.mlp.up_proj.lora_A.weight.weights
      )
      : undefined,
    up_proj_b: (hfAdapterConfig.target_modules as string[]).includes("up_proj")
      ? hfAdapterWeights.base_model.model.model.layers.map((l) =>
        l.mlp.up_proj.lora_B.weight.weights
      )
      : undefined,
  };

  return { adapterConfig, adapterWeights };
}

function permuteReverse(
  win: WasmQ8Tensor,
  n_heads: number,
  dim1: number,
  dim2: number,
) {
  const w = win.dequantize();

  const newShape = [n_heads, 2, Math.floor(dim1 / n_heads / 2), dim2];

  // Reshape w into newShape
  const reshaped: number[][][][] = [];
  let index = 0;
  for (let i = 0; i < newShape[0]; i++) {
    reshaped[i] = [];
    for (let j = 0; j < newShape[1]; j++) {
      reshaped[i][j] = [];
      for (let k = 0; k < newShape[2]; k++) {
        reshaped[i][j][k] = [];
        for (let l = 0; l < newShape[3]; l++) {
          reshaped[i][j][k][l] = w[index++];
        }
      }
    }
  }

  // Transpose (1, 2) => (0, 2, 1, 3)
  const transposed: number[][][][] = [];
  for (let i = 0; i < newShape[0]; i++) {
    transposed[i] = [];
    for (let k = 0; k < newShape[2]; k++) {
      transposed[i][k] = [];
      for (let j = 0; j < newShape[1]; j++) {
        transposed[i][k][j] = reshaped[i][j][k];
      }
    }
  }

  // Flatten the transposed array and reshape it into [dim1, dim2]
  const flattened: number[] = [];
  for (let i = 0; i < newShape[0]; i++) {
    for (let k = 0; k < newShape[2]; k++) {
      for (let j = 0; j < newShape[1]; j++) {
        for (let l = 0; l < newShape[3]; l++) {
          flattened.push(transposed[i][k][j][l]);
        }
      }
    }
  }

  const out = JSQ8Tensor.allocateFrom(flattened, win.gs);
  win.q.array.set(out.q.array);
  win.s.array.set(out.s.array);
}
