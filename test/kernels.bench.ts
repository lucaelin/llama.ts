import { matmul, qmatmul } from "../src/kernels.ts";
import { newQ8Array, quantize } from "../src/quantization.ts";

Deno.bench("matmul perf", (b) => {
  // generate large random arrays
  const n = 4096;
  const d = 4096;
  const x = new Float32Array(n);
  const w = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    x[i] = Math.random();
  }
  for (let i = 0; i < n * d; i++) {
    w[i] = Math.random();
  }

  const o = new Float32Array(d);

  // benchmark
  b.start();
  matmul(o, x, w, n, d);
  b.end();
});

Deno.bench("qmatmul perf", (b) => {
  // generate large random arrays
  const gs = 64;
  const n = 4096;
  const d = 4096;
  const x = new Float32Array(n);
  const w = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    x[i] = Math.random();
  }
  for (let i = 0; i < n * d; i++) {
    w[i] = Math.random();
  }

  // quantize
  const x_q = newQ8Array(n, gs);
  const w_q = newQ8Array(n * d, gs);
  quantize(w_q, w, n * d, gs);

  const o = new Float32Array(d);

  // benchmark
  b.start();
  quantize(x_q, x, n, gs);
  qmatmul(o, x_q, w_q, n, d, gs);
  b.end();
});
