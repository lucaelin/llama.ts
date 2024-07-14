export function accum(a: Float32Array, b: Float32Array, size: number): void {
  for (let i = 0; i < size; i++) a[i] += b[i];
}

export function elemmul(a: Float32Array, b: Float32Array, size: number): void {
  for (let i = 0; i < size; i++) a[i] = a[i] * b[i];
}

export function rmsnorm(
  o: Float32Array,
  x: Float32Array,
  weight: Float32Array,
  size: number,
): void {
  let ss = 0;
  for (let j = 0; j < size; j++) ss += x[j] * x[j];
  ss /= size;
  ss = 1.0 / Math.sqrt(1e-5 + ss);
  for (let j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

export function softmax(x: Float32Array, size: number): void {
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

export function matmul(
  o: Float32Array,
  x: Float32Array,
  w: Float32Array,
  n: number,
  d: number,
): void {
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
export function matmuladd(
  o: Float32Array,
  x: Float32Array,
  w: Float32Array,
  n: number,
  d: number,
  scale: number = 1.0,
): void {
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

export function silu(x: Float32Array, size: number): void {
  // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
  for (let i = 0; i < size; i++) {
    x[i] = x[i] * (1.0 / (1.0 + Math.exp(-x[i])));
  }
}
