import {
  matmul as wasm_matmul,
  qmatmul as wasm_qmatmul,
  quantize as wasm_quantize,
  reset as wasm_reset,
} from "../src/kernels_wasm.ts";
import { WasmF32Tensor, WasmQ8Tensor } from "../src/types_wasm.ts";

const n = 4096;
const d = 2048;

Deno.bench("quantize perf wasm", (b) => {
  wasm_reset();
  // generate large random arrays
  const gs = 64;
  const x = WasmF32Tensor.allocate(n * d);
  for (let i = 0; i < n * d; i++) {
    x.array[i] = Math.random();
  }

  const xq = WasmQ8Tensor.allocate(n * d, gs);

  // benchmark
  b.start();
  wasm_quantize(xq, x, n * d, gs);
  b.end();
});

Deno.bench("matmul perf wasm", (b) => {
  wasm_reset();
  // generate large random arrays
  const x = WasmF32Tensor.allocate(n);
  const w = WasmF32Tensor.allocate(n * d);
  for (let i = 0; i < n; i++) {
    x.array[i] = Math.random();
  }
  for (let i = 0; i < n * d; i++) {
    w.array[i] = Math.random();
  }

  const o = WasmF32Tensor.allocate(d);

  // benchmark
  b.start();
  wasm_matmul(o, x, w, n, d);
  b.end();
});

Deno.bench("qmatmul perf wasm", (b) => {
  wasm_reset();
  // generate large random arrays
  const gs = 64;
  const x = WasmF32Tensor.allocate(n);
  const w = WasmF32Tensor.allocate(n * d);
  for (let i = 0; i < n; i++) {
    x.array[i] = Math.random();
  }
  for (let i = 0; i < n * d; i++) {
    w.array[i] = Math.random();
  }

  // quantize
  const wq = WasmQ8Tensor.allocate(n * d, gs);
  wasm_quantize(wq, w, n * d, gs);

  const xq = WasmQ8Tensor.allocate(n, gs);
  const o = WasmF32Tensor.allocate(d);

  // benchmark
  b.start();
  wasm_quantize(xq, x, n, gs);
  wasm_qmatmul(o, xq, wq, n, d, gs);
  b.end();
});
