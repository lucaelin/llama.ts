import { matmul, matmuladd, softmax } from "./kernels.ts";
import { F32Tensor } from "./types.ts";

export function rope(
  q_t: F32Tensor,
  k_t: F32Tensor,
  pos: number,
  dim: number,
  kv_dim: number,
  head_size: number,
  theta: number,
): void {
  const q = q_t.array;
  const k = k_t.array;

  // RoPE relative positional encoding: complex-valued rotate q and k
  //console.log('RoPE');
  for (let i = 0; i < dim; i += 2) {
    const head_dim = i % head_size;
    const freq = 1.0 / Math.pow(theta, head_dim / head_size);
    const val = pos * freq;
    const fcr = Math.cos(val);
    const fci = Math.sin(val);

    if (i < kv_dim) {
      const v0 = k[i];
      const v1 = k[i + 1];
      k[i] = v0 * fcr - v1 * fci;
      k[i + 1] = v0 * fci + v1 * fcr;
    }

    const v0 = q[i];
    const v1 = q[i + 1];
    q[i] = v0 * fcr - v1 * fci;
    q[i + 1] = v0 * fci + v1 * fcr;
  }
}

export function mutlihead_attention(
  xb_t: F32Tensor,
  q_t: F32Tensor,
  key_cache_layer_t: F32Tensor,
  value_cache_layer_t: F32Tensor,
  att_t: F32Tensor,
  pos: number,
  seq_len: number,
  dim: number,
  n_heads: number,
  n_kv_heads: number,
) {
  const xb = xb_t.array;
  const q = q_t.array;
  const key_cache_layer = key_cache_layer_t.array;
  const value_cache_layer = value_cache_layer_t.array;

  const head_size = dim / n_heads;
  const kv_dim = (dim * n_kv_heads) / n_heads;
  const kv_mul = n_heads / n_kv_heads; // integer multiplier of the kv sharing in multiquery

  //console.log("Multi-Headed Attention");
  for (let h = 0; h < n_heads; h++) {
    const q_head = q.subarray(h * head_size, h * head_size + head_size);
    const att_head_t = att_t.subarray(h * seq_len, h * seq_len + seq_len);
    const att_head = att_head_t.array;

    // iterate over all timesteps, including the current one
    for (let t = 0; t <= pos; t++) {
      // get the key vector for this head and at this timestep
      const k = key_cache_layer.subarray(
        t * kv_dim + Math.floor(h / kv_mul) * head_size,
        t * kv_dim + Math.floor(h / kv_mul) * head_size + head_size,
      );
      // calculate the attention score as the dot product of q and k
      let score = 0.0;
      for (let i = 0; i < head_size; i++) score += q_head[i] * k[i];
      // save the score to the attention buffer
      att_head[t] = score / Math.sqrt(head_size);
    }

    softmax(att_head_t, pos + 1);

    // weighted sum of the values, store back into xb
    const xb_head = xb.subarray(h * head_size, h * head_size + head_size);
    xb_head.fill(0, 0, head_size);
    for (let t = 0; t <= pos; t++) {
      const v = value_cache_layer.subarray(
        t * kv_dim + Math.floor(h / kv_mul) * head_size,
        t * kv_dim + Math.floor(h / kv_mul) * head_size + head_size,
      );
      const att_t = att_head[t];
      for (let i = 0; i < head_size; i++) {
        xb_head[i] += att_t * v[i];
      }
    }
  }
}

export function lora(
  x_out: F32Tensor,
  x: F32Tensor,
  x_r: F32Tensor,
  w_a: F32Tensor,
  w_b: F32Tensor,
  n: number,
  d: number,
  rank: number,
  scale: number,
): void {
  matmul(x_r, x, w_a, n, rank);
  matmuladd(x_out, x_r, w_b, rank, d, scale);
}
