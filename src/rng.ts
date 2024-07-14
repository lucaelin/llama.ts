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

export function random_f32() { // random float32 in [0,1)
  return Math.fround((random_u32() / 256) / 16777216.0);
}
