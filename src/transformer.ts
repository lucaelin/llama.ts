import { matmul, matmuladd, softmax } from "./kernels.ts";

export function rope(
  q: Float32Array,
  k: Float32Array,
  pos: number,
  dim: number,
  kv_dim: number,
  head_size: number,
  theta: number,
): void {
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
  xb: Float32Array,
  q: Float32Array,
  key_cache_layer: Float32Array,
  value_cache_layer: Float32Array,
  att: Float32Array,
  pos: number,
  seq_len: number,
  dim: number,
  n_heads: number,
  n_kv_heads: number,
) {
  const head_size = dim / n_heads;
  const kv_dim = (dim * n_kv_heads) / n_heads;
  const kv_mul = n_heads / n_kv_heads; // integer multiplier of the kv sharing in multiquery

  //console.log("Multi-Headed Attention");
  for (let h = 0; h < n_heads; h++) {
    const q_head = q.subarray(h * head_size, h * head_size + head_size);
    const att_head = att.subarray(h * seq_len, h * seq_len + seq_len);

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

    softmax(att_head, pos + 1);

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
  x_out: Float32Array,
  x: Float32Array,
  x_r: Float32Array,
  w_a: Float32Array,
  w_b: Float32Array,
  n: number,
  d: number,
  rank: number,
  scale: number,
): void {
  matmul(x_r, x, w_a, n, rank);
  matmuladd(x_out, x_r, w_b, rank, d, scale);
}
