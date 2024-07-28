import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { matmul, qmatmul, quantize } from "../src/kernels_wasm.ts";
import { WasmF32Tensor, WasmQ8Tensor } from "../src/types_wasm.ts";

Deno.test("matmul test", () => {
  const o = WasmF32Tensor.allocate(2);
  const x = WasmF32Tensor.allocateFrom([1, 2, 3, 4]);
  const w = WasmF32Tensor.allocateFrom([1, 2, 3, 4, 5, 6, 7, 8]);
  const n = 4;
  const d = 2;
  matmul(o, x, w, n, d);
  assertEquals(o.array, new Float32Array([30, 70]));
});

Deno.test("qmatmul test", () => {
  const gs = 4;
  const o = WasmF32Tensor.allocate(2);
  const x_q = WasmQ8Tensor.allocate(4, gs);
  quantize(x_q, WasmF32Tensor.allocateFrom([1, 2, 3, 4]), 4, gs);
  const w_q = WasmQ8Tensor.allocate(8, gs);
  quantize(w_q, WasmF32Tensor.allocateFrom([1, 2, 3, 4, 5, 6, 7, 8]), 8, gs);
  const n = 4;
  const d = 2;
  qmatmul(o, x_q, w_q, n, d, gs);
  assertEquals(
    o.array,
    new Float32Array([30.0318660736084, 69.9996337890625]),
  );
});
