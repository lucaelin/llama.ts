import { matmul, matmuladd, softmax } from "./kernels.ts";
import { F32Tensor } from "./types.ts";

export function rope(
  q_t: F32Tensor,
  k_t: F32Tensor,
  pos: number,
  num_heads: number,
  num_kv_heads: number,
  head_size: number,
  theta: number,
): void {
  const q = q_t.array;
  const k = k_t.array;
  /*
      for (int i = 0; i < p->n_heads; i++) {
	        for (int j = 0; j < head_size; j += 2) {
	            float freq = 1.0f / powf(500000.0f, (float)j / (float)head_size);
	            float val = pos * freq;
	            float fcr = cosf(val);
	            float fci = sinf(val);
	            float q0 = s->q[i * head_size + j];
	            float q1 = s->q[i * head_size + j + 1];
	            s->q[i * head_size + j] = q0 * fcr - q1 * fci;
	            s->q[i * head_size + j + 1] = q0 * fci + q1 * fcr;
	            if (i < p->n_kv_heads) {
	                float k0 = s->k[i * head_size + j];
	                float k1 = s->k[i * head_size + j + 1];
	                s->k[i * head_size + j] = k0 * fcr - k1 * fci;
	                s->k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
	            }
	        }
	    }
    */

  // RoPE relative positional encoding: complex-valued rotate q and k
  //console.log('RoPE');
  for (let i = 0; i < num_heads; i++) {
    for (let j = 0; j < head_size; j += 2) {
      const freq = 1.0 / Math.pow(theta, j / head_size);
      const val = pos * freq;
      const fcr = Math.cos(val);
      const fci = Math.sin(val);

      if (i < num_kv_heads) {
        const k0 = k[i * head_size + j];
        const k1 = k[i * head_size + j + 1];
        k[i * head_size + j] = k0 * fcr - k1 * fci;
        k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
      }

      const q0 = q[i * head_size + j];
      const q1 = q[i * head_size + j + 1];
      q[i * head_size + j] = q0 * fcr - q1 * fci;
      q[i * head_size + j + 1] = q0 * fci + q1 * fcr;
    }
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
      att_head[t] = score / Math.sqrt(head_size); // gemma2: config.query_pre_attn_scalar**-0.5
    }

    // gemma2 softcap
    //if ( model_type==="gemma2" ) {
    //  attn_weights = attn_weights / self.config.attn_logit_softcapping
    //  attn_weights = torch.tanh(attn_weights)
    //  attn_weights = attn_weights * self.config.attn_logit_softcapping
    //}

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
