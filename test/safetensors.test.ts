import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { bf16ToF32 } from "../src/safetensors.ts";

Deno.test("bf16ToF32 test", () => {
  const a = new Uint8Array([0, 0, 0, 0]);
  assertEquals(bf16ToF32(a), new Float32Array([0, 0]));

  const b = new Uint8Array([0b10000000, 0b00111111, 0b00000000, 0b11000000]);
  assertEquals(bf16ToF32(b), new Float32Array([1, -2]));

  const c = new Uint8Array([0b00000000, 0b00000000, 0b00000000, 0b10000000]);
  assertEquals(bf16ToF32(c), new Float32Array([0, -0]));

  const d = new Uint8Array([0b10000000, 0b01111111, 0b10000000, 0b11111111]);
  assertEquals(bf16ToF32(d), new Float32Array([Infinity, -Infinity]));

  const e = new Uint8Array([0b01001001, 0b01000000, 0b10101011, 0b00111110]);
  //assertEquals(bf16ToF32(d), new Float32Array([Math.PI, 1 / 3]));
  assertEquals(
    bf16ToF32(e),
    new Float32Array([
      3.140625, // Math.PI
      0.333984375, // 1 / 3
    ]),
  );

  const f = new Uint8Array([0xcd, 0x3d, 0x24, 0x3c, 0x83, 0x3a]);
  //assertEquals(bf16ToF32(e), new Float32Array([0.1, 0.01, 0.001]));
  assertEquals(
    bf16ToF32(f),
    new Float32Array([
      0.10009765625, // 0.1
      0.010009765625, // 0.01
      0.00099945068359375, // 0.001
    ]),
  );
});
