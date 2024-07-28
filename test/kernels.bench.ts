import { matmul, qmatmul, quantize } from "../src/kernels.ts";
import { JSF32Tensor, JSQ8Tensor } from "../src/types.ts";

const n = 4096;
const d = 2048;

Deno.bench("quantize perf", (b) => {
  // generate large random arrays
  const gs = 64;
  const x = JSF32Tensor.allocate(n * d);
  for (let i = 0; i < n * d; i++) {
    x.array[i] = Math.random();
  }

  const xq = JSQ8Tensor.allocate(n * d, gs);

  // benchmark
  b.start();
  quantize(xq, x, n * d, gs);
  b.end();
});

Deno.bench("matmul perf", (b) => {
  // generate large random arrays
  const x = JSF32Tensor.allocate(n);
  const w = JSF32Tensor.allocate(n * d);
  for (let i = 0; i < n; i++) {
    x.array[i] = Math.random();
  }
  for (let i = 0; i < n * d; i++) {
    w.array[i] = Math.random();
  }

  const o = JSF32Tensor.allocate(d);

  // benchmark
  b.start();
  matmul(o, x, w, n, d);
  b.end();
});

Deno.bench("qmatmul perf", (b) => {
  // generate large random arrays
  const gs = 64;
  const x = JSF32Tensor.allocate(n);
  const w = JSF32Tensor.allocate(n * d);
  for (let i = 0; i < n; i++) {
    x.array[i] = Math.random();
  }
  for (let i = 0; i < n * d; i++) {
    w.array[i] = Math.random();
  }

  // quantize
  const xq = JSQ8Tensor.allocate(n, gs);
  const wq = JSQ8Tensor.allocate(n * d, gs);
  quantize(wq, w, n * d, gs);

  const o = JSF32Tensor.allocate(d);

  // benchmark
  b.start();
  quantize(xq, x, n, gs);
  qmatmul(o, xq, wq, n, d, gs);
  b.end();
});
