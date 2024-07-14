import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  accum,
  argmax,
  elemmul,
  matmul,
  rmsnorm,
  silu,
  softmax,
} from "../src/kernels.ts";

Deno.test("accum test", () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([4, 5, 6]);
  accum(a, b, 3);
  assertEquals(a, new Float32Array([5, 7, 9]));
});

Deno.test("elemmul test", () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([4, 5, 6]);
  elemmul(a, b, 3);
  assertEquals(a, new Float32Array([4, 10, 18]));
});

Deno.test("rmsnorm test", () => {
  const o = new Float32Array(3);
  const x = new Float32Array([1, 2, 1]);
  const weight = new Float32Array([1, 1, 2]);
  rmsnorm(o, x, weight, 3);
  //const res = Math.sqrt(2);
  //assertEquals(o, new Float32Array([res / 2, res, res]));
  assertEquals(
    o,
    new Float32Array([
      0.7071050405502319,
      1.4142100811004639,
      1.4142100811004639,
    ]),
  );
});

Deno.test("silu test", () => {
  const x = new Float32Array([1, 2, 3]);
  silu(x, 3);
  assertEquals(
    x,
    new Float32Array([
      0.7310585975646973,
      1.7615941762924194,
      2.857722282409668,
    ]),
  );
});

Deno.test("matmul test", () => {
  const o = new Float32Array(2);
  const x = new Float32Array([1, 2, 3]);
  const w = new Float32Array([1, 2, 3, 4, 5, 6]);
  const n = 3;
  const d = 2;
  matmul(o, x, w, n, d);
  assertEquals(o, new Float32Array([14, 32]));
});

Deno.test("softmax test", () => {
  const x = new Float32Array([1, 2, 3]);
  softmax(x, 3);
  assertEquals(
    x,
    new Float32Array([
      0.09003057307052612,
      0.24472847402000427,
      0.6652409439086914,
    ]),
  );
});

Deno.test("argmax test", () => {
  const x = new Float32Array([1, 2, 3]);
  assertEquals(argmax(x, 3), 2);
});
