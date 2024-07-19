import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { dequantize, newQ8Array, quantize } from "../src/quantization.ts";

Deno.test("quantize test", () => {
  const gs = 2;
  const o = newQ8Array(4, gs);
  const x = new Float32Array([0.0001, 0.002, 0.03, 0.4]);

  quantize(o, x, 4, gs);
  assertEquals(o.q, new Int8Array([6, 127, 10, 127]));
  assertEquals(
    o.s,
    new Float32Array([0.000015748031728435308, 0.0031496062874794006]),
  );

  const dq = new Float32Array(4);
  dequantize(dq, o, 4, gs);
  assertEquals(
    dq,
    new Float32Array([
      0.00009448819037061185,
      0.002000000094994902,
      0.031496062874794006,
      0.4000000059604645,
    ]),
  );
});
