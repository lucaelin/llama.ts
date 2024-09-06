import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  accum,
  argmax,
  dequantize,
  elemmul,
  matmul,
  qmatmul,
  quantize,
  rmsnorm,
  silu,
  softmax,
} from "../src/kernels.ts";
import { JSF32Tensor, JSQ8Tensor } from "../src/types.ts";

Deno.test("accum test", () => {
  const a = JSF32Tensor.allocateFrom([1, 2, 3]);
  const b = JSF32Tensor.allocateFrom([4, 5, 6]);
  accum(a, b, 3);
  assertEquals(a.array, new Float32Array([5, 7, 9]));
});

Deno.test("elemmul test", () => {
  const a = JSF32Tensor.allocateFrom([1, 2, 3]);
  const b = JSF32Tensor.allocateFrom([4, 5, 6]);
  elemmul(a, b, 3);
  assertEquals(a.array, new Float32Array([4, 10, 18]));
});

Deno.test("rmsnorm test", () => {
  const o = JSF32Tensor.allocate(3);
  const x = JSF32Tensor.allocateFrom([1, 2, 1]);
  const weight = JSF32Tensor.allocateFrom([1, 1, 2]);
  rmsnorm(o, x, weight, 3, 0);
  //const res = Math.sqrt(2);
  //assertEquals(o, new Float32Array([res / 2, res, res]));
  assertEquals(
    o.array,
    new Float32Array([
      0.7071050405502319,
      1.4142100811004639,
      1.4142100811004639,
    ]),
  );
});

Deno.test("silu test", () => {
  const x = JSF32Tensor.allocateFrom([1, 2, 3]);
  silu(x, 3);
  assertEquals(
    x.array,
    new Float32Array([
      0.7310585975646973,
      1.7615941762924194,
      2.857722282409668,
    ]),
  );
});

Deno.test("matmul test", () => {
  const o = JSF32Tensor.allocate(2);
  const x = JSF32Tensor.allocateFrom([1, 2, 3, 4]);
  const w = JSF32Tensor.allocateFrom([1, 2, 3, 4, 5, 6, 7, 8]);
  const n = 4;
  const d = 2;
  matmul(o, x, w, n, d);
  assertEquals(o.array, new Float32Array([30, 70]));
});

Deno.test("softmax test", () => {
  const x = JSF32Tensor.allocateFrom([1, 2, 3]);
  softmax(x, 3);
  assertEquals(
    x.array,
    new Float32Array([
      0.09003057307052612,
      0.24472847402000427,
      0.6652409439086914,
    ]),
  );
});

Deno.test("argmax test", () => {
  const x = JSF32Tensor.allocateFrom([1, 2, 3]);
  assertEquals(argmax(x, 3), 2);
});

Deno.test("qmatmul test", () => {
  const gs = 4;
  const o = JSF32Tensor.allocate(2);
  const x_q = JSQ8Tensor.allocate(4, gs);
  quantize(x_q, JSF32Tensor.allocateFrom([1, 2, 3, 4]), 4, gs);
  const w_q = JSQ8Tensor.allocate(8, gs);
  quantize(w_q, JSF32Tensor.allocateFrom([1, 2, 3, 4, 5, 6, 7, 8]), 8, gs);
  const n = 4;
  const d = 2;
  qmatmul(o, x_q, w_q, n, d, gs);
  assertEquals(
    o.array,
    new Float32Array([30.03186798095703, 69.99962615966797]),
  );
});

Deno.test("quantize test", () => {
  const gs = 2;
  const o = JSQ8Tensor.allocate(4, gs);
  const x = JSF32Tensor.allocateFrom([0.0001, 0.002, 0.03, 0.4]);

  quantize(o, x, 4, gs);
  assertEquals(o.q.array, new Int8Array([6, 127, 10, 127]));
  assertEquals(
    o.s.array,
    new Float32Array([0.000015748031728435308, 0.0031496062874794006]),
  );

  const dq = JSF32Tensor.allocate(4);
  dequantize(dq, o, 4, gs);
  assertEquals(
    dq.array,
    new Float32Array([
      0.00009448819037061185,
      0.002000000094994902,
      0.031496062874794006,
      0.4000000059604645,
    ]),
  );
});
