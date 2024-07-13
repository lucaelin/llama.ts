let rng_seed: bigint = 0n;

export function set_seed(seed: number) {
  rng_seed = BigInt(seed);
}

export function random_u32(): number {
  rng_seed ^= rng_seed >> 12n;
  rng_seed ^= (rng_seed << 25n) & 0xffffffffffffffffn;
  rng_seed ^= rng_seed >> 27n;
  return Number(((rng_seed * 0x2545F4914F6CDD1Dn) >> 32n) & 0xffffffffn);
}

const floatCaster = new Float32Array(1);
export function random_f32() { // random float32 in [0,1)
  floatCaster[0] = (random_u32() / 256) / 16777216.0;
  return floatCaster[0]; // force f32
}
