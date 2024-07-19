export const DEFAULT_GROUP_SIZE = 32; // Andrej uses 64, according to thebloke 128 is optimal, the current implementation degrades above 32 for small models

export class Q8Array {
  public readonly gs: number;
  constructor(public readonly q: Int8Array, public readonly s: Float32Array) {
    this.gs = q.length / s.length;
  }

  public get length(): number {
    return this.q.length;
  }

  public slice(start: number, end: number): Q8Array {
    if (start % this.gs !== 0 || end % this.gs !== 0) {
      throw new Error("Slice start and end must be a multiple of group size");
    }
    return new Q8Array(
      this.q.slice(start, end),
      this.s.slice(start / this.gs, end / this.gs),
    );
  }

  public subarray(start: number, end: number): Q8Array {
    if (start % this.gs !== 0 || end % this.gs !== 0) {
      throw new Error(
        "Subarray start and end must be a multiple of group size",
      );
    }
    return new Q8Array(
      this.q.subarray(start, end),
      this.s.subarray(start / this.gs, end / this.gs),
    );
  }
}

export function newQ8Array(n: number, gs: number): Q8Array {
  return new Q8Array(new Int8Array(n), new Float32Array(n / gs));
}

export function newQ8ArrayFrom(
  x: Float32Array,
  gs: number,
): Q8Array {
  if (x.length % gs !== 0) {
    throw new Error("Input length must be a multiple of group size");
  }
  const o = newQ8Array(x.length, gs);
  quantize(o, x, x.length, gs);
  return o;
}

export function q8ArrayToFloat32Array(
  x: Q8Array,
  gs: number,
): Float32Array {
  const n = x.q.length;
  const o = new Float32Array(n);
  dequantize(o, x, n, gs);
  return o;
}

export function dequantize(
  o: Float32Array,
  x: Q8Array,
  n: number,
  gs: number,
): void {
  for (let i = 0; i < n; i++) {
    o[i] = x.q[i] * x.s[Math.floor(i / gs)];
  }
}

export function quantize(
  o: Q8Array,
  x: Float32Array,
  n: number,
  gs: number,
): void {
  const num_groups = n / gs;
  const Q_MAX = 127.0;
  let err = 0.0;

  if (o.length !== n || x.length !== n) {
    throw new Error("Length mismatch");
  }
  if (x.length % gs !== 0) {
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
    o.s[group] = scale;

    // calculate and write the quantized values
    for (let i = 0; i < gs; i++) {
      const original_value = x[group * gs + i];
      const quant_value = original_value / scale; // scale
      const quantized = Math.round(quant_value); // round and clamp
      o.q[group * gs + i] = quantized;

      const restored_value = quantized * scale;
      err += Math.abs(original_value - restored_value);
    }
  }

  //console.log("Quantization error: ", err / n);
}
