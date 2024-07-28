import { F32Tensor, Q8Tensor } from "./types.ts";

export function accum(a_t: F32Tensor, b_t: F32Tensor, size: number): void {
  const a = a_t.array;
  const b = b_t.array;
  for (let i = 0; i < size; i++) a[i] += b[i];
}

export function elemmul(
  a_t: F32Tensor,
  b_t: F32Tensor,
  size: number,
): void {
  const a = a_t.array;
  const b = b_t.array;
  for (let i = 0; i < size; i++) a[i] = a[i] * b[i];
}

export function rmsnorm(
  o_t: F32Tensor,
  x_t: F32Tensor,
  weight_t: F32Tensor,
  size: number,
): void {
  const o = o_t.array;
  const x = x_t.array;
  const weight = weight_t.array;

  let ss = 0;
  for (let j = 0; j < size; j++) ss += x[j] * x[j];
  ss /= size;
  ss = 1.0 / Math.sqrt(1e-5 + ss);
  for (let j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

export function softmax(x_t: F32Tensor, size: number): void {
  const x = x_t.array;

  // find max value (for numerical stability)
  let max_val = x[0];
  for (let i = 1; i < size; i++) {
    if (x[i] > max_val) max_val = x[i];
  }

  // exp and sum
  let sum = 0;
  for (let i = 0; i < size; i++) {
    x[i] = Math.exp(x[i] - max_val);
    sum += x[i];
  }

  // normalize
  for (let i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

export function argmax(arr_t: F32Tensor, size: number): number {
  const arr = arr_t.array;

  let max_val = arr[0];
  let max_idx = 0;
  for (let i = 1; i < size; i++) {
    if (arr[i] > max_val) {
      max_val = arr[i];
      max_idx = i;
    }
  }
  return max_idx;
}

export function matmul(
  o_t: F32Tensor,
  x_t: F32Tensor,
  w_t: F32Tensor,
  n: number,
  d: number,
): void {
  const o = o_t.array;
  const x = x_t.array;
  const w = w_t.array;

  // W (d, n) @ x (n,) -> xout (d,)
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      //if (i * n + j >= w.length) throw new Error("matmul weights out of bounds");
      //if (j >= x.length) throw new Error("matmul activations out of bounds");
      sum += w[i * n + j | 0] * x[j];
    }
    //if (i >= o.length) throw new Error("matmul output out of bounds");
    o[i] = sum;
  }
}

export function qmatmul(
  o_t: F32Tensor,
  x_t: Q8Tensor,
  w_t: Q8Tensor,
  n: number,
  d: number,
  GS: number,
): void {
  const o = o_t.array;

  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  // inputs to this function are both quantized

  const xq = x_t.q.array;
  const xs = x_t.s.array;
  const wq = w_t.q.array;
  const ws = w_t.s.array;

  if (x_t.gs !== GS || w_t.gs !== GS) {
    throw new Error("Group size mismatch");
  }
  if (xq.length !== n || wq.length !== n * d) {
    throw new Error("Quantized array size mismatch");
  }

  for (let i = 0; i < d; i++) {
    let val = 0.0;
    const itn = i * n;

    // do the matmul in groups of GS
    for (let j = 0; j <= n - GS; j += GS) {
      let ival = 0;
      for (let k = 0; k < GS; k++) {
        ival += xq[j + k] * wq[itn + j + k];
      }
      const bar = ival * ws[((itn + j) / GS) | 0];
      val += bar * xs[(j / GS) | 0];
    }

    o[i] = val;
  }
}

export function matmuladd(
  o_t: F32Tensor,
  x_t: F32Tensor,
  w_t: F32Tensor,
  n: number,
  d: number,
  scale: number = 1.0,
): void {
  const o = o_t.array;
  const x = x_t.array;
  const w = w_t.array;

  // W (d, n) @ x (n,) -> xout (d,)
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      //if (i * n + j >= w.length) throw new Error("matmul weights out of bounds");
      //if (j >= x.length) throw new Error("matmul activations out of bounds");
      sum += w[i * n + j] * x[j];
    }
    //if (i >= xout.length) throw new Error("matmul output out of bounds");
    o[i] += sum * scale;
  }
}

export function qmatmuladd(
  o_t: F32Tensor,
  x_t: Q8Tensor,
  w_t: Q8Tensor,
  n: number,
  d: number,
  GS: number,
  scale: number = 1.0,
): void {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  // inputs to this function are both quantized
  const o = o_t.array;

  const xq = x_t.q.array;
  const xs = x_t.s.array;
  const wq = w_t.q.array;
  const ws = w_t.s.array;

  if (x_t.gs !== GS || w_t.gs !== GS) {
    throw new Error("Group size mismatch");
  }
  if (x_t.q.length !== n || w_t.q.length !== n * d) {
    throw new Error("Quantized array size mismatch");
  }

  for (let i = 0; i < d; i++) {
    let val = 0.0;
    const itn = i * n;

    // do the matmul in groups of GS
    for (let j = 0; j <= n - GS; j += GS) {
      let ival = 0;
      for (let k = 0; k < GS; k++) {
        ival += xq[j + k] * wq[itn + j + k];
      }
      const bar = ival * ws[((itn + j) / GS) | 0];
      val += bar * xs[(j / GS) | 0];
    }

    o[i] += val * scale;
  }
}

export function silu(x_t: F32Tensor, size: number): void {
  const x = x_t.array;

  // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
  for (let i = 0; i < size; i++) {
    x[i] = x[i] * (1.0 / (1.0 + Math.exp(-x[i])));
  }
}

export function dequantize(
  o_t: F32Tensor,
  x_t: Q8Tensor,
  n: number,
  gs: number,
): void {
  const o = o_t.array;
  const xq = x_t.q.array;
  const xs = x_t.s.array;
  for (let i = 0; i < n; i++) {
    o[i] = xq[i] * xs[Math.floor(i / gs)];
  }
}

export function quantize(
  o_t: Q8Tensor,
  x_t: F32Tensor,
  n: number,
  gs: number,
): void {
  const oq = o_t.q.array;
  const os = o_t.s.array;
  const x = x_t.array;

  const num_groups = n / gs;
  const Q_MAX = 127.0;
  let err = 0.0;

  if (o_t.length !== n || x_t.length !== n) {
    throw new Error("Length mismatch");
  }
  if (x_t.length % gs !== 0) {
    throw new Error("Input length must be a multiple of group size");
  }

  for (let group = 0; group < num_groups; group++) {
    // find the max absolute value in the current group
    let wmax = 0.0;
    for (let i = 0; i < gs; i++) {
      const val = Math.abs(x[group * gs + i]);
      if (val > wmax) {
        wmax = val;
      }
    }

    // calculate and write the scaling factor
    const scale = wmax / Q_MAX;
    os[group] = scale;

    // calculate and write the quantized values
    for (let i = 0; i < gs; i++) {
      const original_value = x[group * gs + i];
      const quant_value = original_value / scale; // scale
      const quantized = Math.round(quant_value); // round and clamp
      oq[group * gs + i] = quantized;

      const restored_value = quantized * scale;
      err += Math.abs(original_value - restored_value);
    }
  }

  //console.log("Quantization error: ", err / n);
}
