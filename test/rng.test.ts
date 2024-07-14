import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { random_f32, random_u32, set_seed } from "../src/rng.ts";

Deno.test("random_u32 test", () => {
  set_seed(1);
  assertEquals(random_u32(), 1206177355);
  assertEquals(random_u32(), 2882512552);
  set_seed(2);
  assertEquals(random_u32(), 2412354711);
  assertEquals(random_u32(), 1470057809);
});

Deno.test("random_f32 test", () => {
  set_seed(1);
  assertEquals(random_f32(), 0.2808350622653961);
  assertEquals(random_f32(), 0.671137273311615);
});
