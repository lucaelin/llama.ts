// deno-lint-ignore-file prefer-const
// Llama2 transformer model inference in one TypeScript file.
// by Oleksandr Nikitin, 2023 (MIT licensed).
// Based on the Andrej Karpathy's llama2.c: https://github.com/karpathy/llama2.c
//
// Use bun or t348 to run. see params at the end of the file or in the README.
import * as fs from "node:fs";
import { Buffer } from "node:buffer";
import { writeAllSync } from "https://deno.land/std/io/mod.ts";

// ----------------------------------------------------------------------------
// binary utils

type float = number;
type int = number;
const f32bytes = 4;
const i32bytes = 4;
class BufferReader {
  view: DataView;
  position: number;
  constructor(buffer: Buffer) {
    this.view = new DataView(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength,
    );
    this.position = 0;
  }

  getInt32LE(): int {
    let value = this.view.getInt32(this.position, true);
    this.position += i32bytes;
    return value;
  }

  getFloat32LE(): float {
    let value = this.view.getFloat32(this.position, true);
    this.position += f32bytes;
    return value;
  }

  getBytesInto(bytes: Uint8Array) {
    bytes.set(new Uint8Array(this.view.buffer, this.position, bytes.length));
    this.position += bytes.length;
    return bytes;
  }
}

class FileHandleReader {
  handle: number;
  position: number;
  constructor(handle: number, offset: number) {
    this.handle = handle;
    this.position = offset;
  }
  getF32Array(...dims: number[]): Float32Array {
    let totalFloats = dims.reduce((a, b) => a * b);
    //    console.log({offset, dims, totalBytes, bb:this.view.buffer.length})
    let bytes = Buffer.alloc(totalFloats * f32bytes);
    fs.readSync(this.handle, bytes, 0, bytes.length, this.position);
    let ret = new Float32Array(bytes.buffer, bytes.byteOffset, totalFloats);
    this.position += totalFloats * f32bytes;
    return ret;
  }

  getF32Arrays(dim0: number, ...dims: number[]): Float32Array[] {
    let array = new Array(dim0);
    for (let i = 0; i < dim0; ++i) {
      array[i] = this.getF32Array(...dims);
    }
    return array;
  }
}
interface Config {
  dim: int;
  hidden_dim: int;
  n_layers: int;
  n_heads: int;
  n_kv_heads: int;
  vocab_size: int;
  seq_len: int;
  shared_weights: boolean;
  head_size: int;
}
function readConfig(buffer: BufferReader): Config {
  let c = {} as Config;
  c.dim = buffer.getInt32LE();
  c.hidden_dim = buffer.getInt32LE();
  c.n_layers = buffer.getInt32LE();
  c.n_heads = buffer.getInt32LE();
  c.n_kv_heads = buffer.getInt32LE();
  let vocab_size = buffer.getInt32LE();
  c.vocab_size = Math.abs(vocab_size);
  c.seq_len = buffer.getInt32LE();
  c.shared_weights = vocab_size > 0;
  c.head_size = c.dim / c.n_heads;
  return c;
}

interface TransformerWeights {
  token_embedding_table: Float32Array;
  rms_att_weight: Float32Array[];
  wq: Float32Array[];
  wk: Float32Array[];
  wv: Float32Array[];
  wo: Float32Array[];
  rms_ffn_weight: Float32Array[];
  w1: Float32Array[];
  w2: Float32Array[];
  w3: Float32Array[];
  rms_final_weight: Float32Array;
  freq_cis_real: Float32Array;
  freq_cis_imag: Float32Array;
  wcls: Float32Array;
}

function readWeights(
  config: Config,
  buffer: FileHandleReader,
  shared_weights: boolean,
): TransformerWeights {
  let w = {} as TransformerWeights;
  console.log(
    config.n_layers,
    config.dim,
    config.n_heads,
    config.n_kv_heads,
    config.head_size,
  );
  w.token_embedding_table = buffer.getF32Array(config.vocab_size, config.dim);
  console.log("w.token_embedding_table[0]: ", w.token_embedding_table[0]);
  w.rms_att_weight = buffer.getF32Arrays(config.n_layers, config.dim);
  console.log("w.rms_att_weight[0][0]: %f", w.rms_att_weight[1][0]);
  w.wq = buffer.getF32Arrays(
    config.n_layers,
    config.dim,
    config.n_heads * config.head_size,
  );
  console.log("w.wq[0][0]: %f", w.wq[1][0]);
  w.wk = buffer.getF32Arrays(
    config.n_layers,
    config.dim,
    config.n_kv_heads * config.head_size,
  );
  console.log("w.wk[0][0]: %f", w.wk[1][0]);
  w.wv = buffer.getF32Arrays(
    config.n_layers,
    config.dim,
    config.n_kv_heads * config.head_size,
  );
  console.log("w.wv[0][0]: %f", w.wv[1][0]);
  w.wo = buffer.getF32Arrays(
    config.n_layers,
    config.n_heads * config.head_size,
    config.dim,
  );
  console.log("w.wo[0][0]: %f", w.wo[1][0]);
  w.rms_ffn_weight = buffer.getF32Arrays(config.n_layers, config.dim); // jagged pointer arithmetic lol
  console.log("w.rms_ffn_weight[0][0]: %f", w.rms_ffn_weight[1][0]);
  w.w1 = buffer.getF32Arrays(config.n_layers, config.hidden_dim, config.dim);
  w.w2 = buffer.getF32Arrays(config.n_layers, config.dim, config.hidden_dim);
  w.w3 = buffer.getF32Arrays(config.n_layers, config.hidden_dim, config.dim);
  w.rms_final_weight = buffer.getF32Array(config.dim);
  w.freq_cis_real = buffer.getF32Array(config.seq_len, config.head_size / 2);
  w.freq_cis_imag = buffer.getF32Array(config.seq_len, config.head_size / 2);
  w.wcls = shared_weights
    ? w.token_embedding_table
    : buffer.getF32Array(config.vocab_size, config.dim);
  return w;
}

interface RunState {
  // current wave of activations
  x: Float32Array;
  xb: Float32Array;
  xb2: Float32Array;
  hb: Float32Array;
  hb2: Float32Array;
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  att: Float32Array; // buffer for scores/attention values (n_heads, seq_len)
  logits: Float32Array;
  key_cache: Float32Array;
  value_cache: Float32Array;
  indices: { prob: float; index: int }[];
}
function newRunState(config: Config): RunState {
  const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
  const s = {} as RunState;
  s.indices = new Array(config.vocab_size);
  s.x = new Float32Array(config.dim);
  s.xb = new Float32Array(config.dim);
  s.xb2 = new Float32Array(config.dim);
  s.hb = new Float32Array(config.hidden_dim);
  s.hb2 = new Float32Array(config.hidden_dim);
  s.q = new Float32Array(config.dim);
  s.k = new Float32Array(config.dim);
  s.v = new Float32Array(config.dim);
  s.att = new Float32Array(config.n_heads * config.seq_len);
  s.logits = new Float32Array(config.vocab_size);
  s.key_cache = new Float32Array(config.n_layers * config.seq_len * kv_dim);
  s.value_cache = new Float32Array(
    config.n_layers * config.seq_len * kv_dim,
  );
  return s;
}

// ----------------------------------------------------------------------------
// neural net blocks

function accum(a: Float32Array, b: Float32Array, size: number): void {
  for (let i = 0; i < size; i++) a[i] += b[i];
}

function rmsnorm(
  o: Float32Array,
  x: Float32Array,
  weight: Float32Array,
  size: number,
): void {
  let ss = 0;
  for (let j = 0; j < size; j++) ss += x[j] * x[j];
  ss /= size;
  ss = 1.0 / Math.sqrt(1e-5 + ss);
  for (let j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
  // debugger;
}

function softmax(x: Float32Array, size: number): void {
  // find max value (for numerical stability)
  let max_val = x[0];
  for (let i = 1; i < size; i++) {
    if (x[i] > max_val) max_val = x[i];
  }

  // exp and sum
  let sum = 0;
  for (let i = 0; i < size; i++) {
    x[i] = Math.exp(x[i] - max_val);
    sum += x[i];
  }

  // normalize
  for (let i = 0; i < size; i++) {
    x[i] /= sum; //Accumulator[0]; // ah forget it, it's numerically stable enough
  }
}

function matmul(
  xout: Float32Array,
  x: Float32Array,
  w: Float32Array,
  n: number,
  d: number,
): void {
  // W (d, n) @ x (n,) -> xout (d,)
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += w[i * n + j] * x[j];
    }
    xout[i] = sum; //sumAccumulator[0];
  }
}

function forward(
  token: number,
  pos: number,
  p: Config,
  s: RunState,
  w: TransformerWeights,
): void {
  const x = s.x;
  const dim = p.dim;
  const kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
  const kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

  const hidden_dim = p.hidden_dim;
  const head_size = dim / p.n_heads;

  // copy the token embedding into x
  x.set(w.token_embedding_table.subarray(token * dim, token * dim + dim));

  //debugger;
  // forward all the layers
  for (let l = 0; l < p.n_layers; l++) {
    //console.log("Layer %d", l + 1);
    // attention rmsnorm
    rmsnorm(s.xb, x, w.rms_att_weight[l], dim);

    // key and value point to the kv cache
    const loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
    s.k = s.key_cache.subarray(
      loff + pos * kv_dim,
      loff + pos * kv_dim + kv_dim,
    );
    s.v = s.value_cache.subarray(
      loff + pos * kv_dim,
      loff + pos * kv_dim + kv_dim,
    );

    // qkv matmuls for this position
    matmul(s.q, s.xb, w.wq[l], dim, dim);
    matmul(s.k, s.xb, w.wk[l], dim, kv_dim);
    matmul(s.v, s.xb, w.wv[l], dim, kv_dim);
    //console.log("s.q[0]: %f", s.q[0]);
    //console.log("s.k[0]: %f", s.k[0]);
    //console.log("s.v[0]: %f", s.v[0]);

    // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
    for (let i = 0; i < dim; i += 2) {
      const head_dim = i % head_size;
      const freq = 1.0 / Math.pow(10000.0, head_dim / head_size);
      const val = pos * freq;
      const fcr = Math.cos(val);
      const fci = Math.sin(val);

      const rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
      for (let v = 0; v < rotn; v++) {
        const vec = v == 0 ? s.q : s.k; // the vector to rotate (query or key)
        const v0 = vec[i];
        const v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    // multihead attention. iterate over all heads
    for (let h = 0; h < p.n_heads; h++) {
      let q = s.q.subarray(h * head_size, h * head_size + head_size);
      let att = s.att.subarray(h * p.seq_len, h * p.seq_len + p.seq_len);

      // iterate over all timesteps, including the current one
      for (let t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        const k = s.key_cache.subarray(
          loff + t * kv_dim + Math.floor(h / kv_mul) * head_size,
          loff + t * kv_dim + Math.floor(h / kv_mul) * head_size + head_size,
        );
        // calculate the attention score as the dot product of q and k
        let score = 0.0;
        for (let i = 0; i < head_size; i++) score += q[i] * k[i];
        // save the score to the attention buffer
        att[t] = score / Math.sqrt(head_size);
      }

      softmax(att, pos + 1);

      // weighted sum of the values, store back into xb
      const xb = s.xb.subarray(h * head_size, h * head_size + head_size);
      xb.fill(0, 0, head_size);
      for (let t = 0; t <= pos; t++) {
        const v = s.value_cache.subarray(
          loff + t * kv_dim + Math.floor(h / kv_mul) * head_size,
          loff + t * kv_dim + Math.floor(h / kv_mul) * head_size + head_size,
        );
        const att_t = att[t];
        for (let i = 0; i < head_size; i++) {
          xb[i] += att_t * v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    matmul(s.xb2, s.xb, w.wo[l], dim, dim);

    // residual connection back into x
    accum(x, s.xb2, dim);

    // ffn rmsnorm
    rmsnorm(s.xb, x, w.rms_ffn_weight[l], dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s.hb, s.xb, w.w1[l], dim, hidden_dim);
    matmul(s.hb2, s.xb, w.w3[l], dim, hidden_dim);

    // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    for (let i = 0; i < hidden_dim; i++) {
      s.hb[i] = s.hb[i] * (1.0 / (1.0 + Math.exp(-s.hb[i])));
    }

    // elementwise multiply with w3(x)
    for (let i = 0; i < hidden_dim; i++) s.hb[i] = s.hb[i] * s.hb2[i];

    // final matmul to get the output of the ffn
    matmul(s.xb, s.hb, w.w2[l], hidden_dim, dim);

    // residual connection
    accum(x, s.xb, dim);
  }

  // final rmsnorm
  rmsnorm(x, x, w.rms_final_weight, dim);

  // classifier into logits
  matmul(s.logits, x, w.wcls, p.dim, p.vocab_size);
}

function encode(
  text: string,
  bos: boolean,
  eos: boolean,
  vocab: string[],
  vocab_scores: number[],
  vocab_size: number,
  tokens: Int32Array,
) {
  // first encode every individual byte in the input string
  let n_tokens = 0; // the number of tokens
  if (bos) tokens[n_tokens++] = 1; // BOS token
  for (let i = 0; i < text.length; ++i) {
    let id = vocab.indexOf(text.charAt(i));
    if (id == -1) {
      throw new Error("Error: character not found in vocab: " + text.charAt(i));
    }
    tokens[n_tokens++] = id;
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (true) {
    let best_score = -1e10;
    let best_id = -1;
    let best_idx = -1;

    for (let i = 0; i < n_tokens - 1; ++i) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      let str_buffer = vocab[tokens[i]] + vocab[tokens[i + 1]];
      let id = vocab.indexOf(str_buffer);
      if (id != -1 && vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) break; // we couldn't find any more pairs to merge, so we're done

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (let i = best_idx + 1; i < n_tokens - 1; i++) {
      tokens[i] = tokens[i + 1];
    }
    n_tokens--; // token length decreased
  }

  if (eos) tokens[n_tokens++] = 1; // EOS token

  return n_tokens;
}

// ----------------------------------------------------------------------------
// utilities: time / rng
let rng_seed: bigint = 0n;
function random_u32(): number {
  rng_seed ^= rng_seed >> 12n;
  rng_seed ^= (rng_seed << 25n) & 0xffffffffffffffffn;
  rng_seed ^= rng_seed >> 27n;
  return Number(((rng_seed * 0x2545F4914F6CDD1Dn) >> 32n) & 0xffffffffn);
}

const floatCaster = new Float32Array(1);
function random_f32() { // random float32 in [0,1)
  floatCaster[0] = (random_u32() / 256) / 16777216.0;
  return floatCaster[0]; // force f32
}

// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
function argmax(arr: Float32Array): number {
  return arr.reduce(
    (maxIdx, val, idx, array) => (val > array[maxIdx] ? idx : maxIdx),
    0,
  );
}

function sample(
  temperature: number,
  state: RunState,
  config: Config,
  topp: number,
) {
  // sample the token given the logits and some hyperparameters
  if (temperature == 0.0) {
    // greedy argmax sampling: take the token with the highest probability
    return argmax(state.logits);
  } else {
    // apply the temperature to the logits
    for (let q = 0; q < config.vocab_size; q++) {
      state.logits[q] /= temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(state.logits, config.vocab_size);
    const coin = random_f32();
    // we sample from this distribution to get the next token
    if (topp <= 0 || topp >= 1) {
      // simply sample from the predicted probability distribution
      return sample_mult(state.logits, config.vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      return sample_topp(
        state.logits,
        config.vocab_size,
        topp,
        state.indices,
        coin,
      );
    }
  }
}

function sample_mult(
  logits: Float32Array,
  n: float,
  coin: float,
): number {
  let cdf = 0;
  for (let i = 0; i < n; i++) {
    cdf += logits[i];
    if (coin < cdf) return i;
  }
  return n - 1;
}

function sample_topp(
  probabilities: Float32Array,
  n: float,
  topp: number,
  probindex: { prob: float; index: int }[],
  coin: float,
): number {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  let n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting

  const cutoff = (1.0 - topp) / (n - 1);
  for (let i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0++] = { index: i, prob: probabilities[i] };
    }
  }
  probindex.sort((a, b) => b.prob - a.prob);

  // truncate the list where cumulative probability exceeds topp
  let cumulativeProb = 0;
  let lastIdx = n0 - 1; // in case of rounding errors consider all elements
  for (let i = 0; i < n; i++) {
    cumulativeProb += probindex[i].prob;
    if (cumulativeProb > topp) {
      lastIdx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  const r = coin * cumulativeProb;
  cumulativeProb = 0;
  for (let i = 0; i < lastIdx; i++) {
    cumulativeProb += probindex[i].prob;
    if (r < cumulativeProb) return probindex[i].index;
  }
  return probindex[lastIdx].index;
}

function decode(vocab: string[], prev_token: number, token: number): string {
  //console.log("token: %d %d", prev_token, token);
  let piece = vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece.charAt(0) == " ") piece = piece.substring(1);

  return piece;
}

// ----------------------------------------------------------------------------
// int main
function main() {
  // defaults
  //console.log(Deno.args);
  const [checkpoint, ...args] = Deno.args;
  let temperature = 1.0; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  let topp = 1.0; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  rng_seed = 0n; // seed rng with time by default
  let steps = 256; // max number of steps to run for, 0: use seq_len
  let prompt: string | null = null; // prompt string

  if (!checkpoint) return error_usage();
  for (let i = 0; i < args.length; i += 2) {
    if (i + 1 >= args.length) return error_usage(); // must have arg after flag
    let [arg, val] = [args[i], args[i + 1]];
    if (arg.charAt(0) != "-") return error_usage(); // must start with dash
    if (arg.length != 2) return error_usage(); // must be -x (one dash, one letter)
    // read in the args
    switch (args[i][1]) {
      case "t":
        temperature = parseFloat(val);
        break;
      case "p":
        topp = parseFloat(val);
        break;
      case "s":
        rng_seed = BigInt(parseInt(val));
        break;
      case "n":
        steps = parseInt(val);
        break;
      case "i":
        prompt = val;
        break;
      default:
        return error_usage();
    }
  }
  if (rng_seed == 0n) rng_seed = BigInt(Date.now());

  // read in the model.bin file
  let fileHandle = fs.openSync(checkpoint, "r");
  let configSize = 7 * i32bytes;

  // read in the config header
  let configBuffer = Buffer.alloc(configSize);
  fs.readSync(fileHandle, configBuffer, 0, configSize, 0);
  let config = readConfig(new BufferReader(configBuffer));
  //console.error(config);
  let weights = readWeights(
    config,
    new FileHandleReader(fileHandle, configSize),
    config.shared_weights,
  );
  fs.closeSync(fileHandle);

  // right now we cannot run for more than config.seq_len steps
  if (steps <= 0 || steps > config.seq_len) steps = config.seq_len;

  // read in the tokenizer.bin file
  let vocab = new Array<string>(config.vocab_size);
  let vocab_scores = new Array<number>(config.vocab_size);
  let tokBuffer = new BufferReader(fs.readFileSync("tokenizer.bin"));
  let _ignored_max_token_length = tokBuffer.getInt32LE();
  for (let i = 0; i < config.vocab_size; i++) {
    vocab_scores[i] = tokBuffer.getFloat32LE();
    vocab[i] = new TextDecoder().decode(
      tokBuffer.getBytesInto(new Uint8Array(tokBuffer.getInt32LE())),
    );
  }
  // create and init the application RunState
  let state = newRunState(config);
  if (prompt == null) prompt = "";

  // encode the (string) prompt into tokens sequence
  let num_prompt_tokens = 0;
  let prompt_tokens: Int32Array = new Int32Array(prompt.length + 3); // +3 for '\0', ?BOS, ?EOS

  num_prompt_tokens = encode(
    prompt,
    true, // bos
    false, // no eos
    vocab,
    vocab_scores,
    config.vocab_size,
    prompt_tokens,
  );
  console.log(
    "Prompt tokens: %d %d %d",
    prompt_tokens[0],
    prompt_tokens[1],
    prompt_tokens[2],
  );

  // start the main loop
  let start = 0; // used to time our code, only initialized after first iteration
  let next; // will store the next token in the sequence
  let token = prompt_tokens[0]; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
  let pos = 0; // position in the sequence
  while (pos < steps) {
    //console.log("Step %d", pos + 1);
    // forward the transformer to get logits for the next token
    forward(token, pos, config, state, weights);

    //console.log("Step %d decoding", pos + 1);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // sample the next token
      next = sample(temperature, state, config, topp);
    }
    pos++;

    // data-dependent terminating condition: the BOS (1) token delimits sequences
    if (next == 1) break;

    // print the token as string, decode it with the Tokenizer object
    const piece = decode(vocab, token, next);

    writeAllSync(Deno.stdout, new TextEncoder().encode(piece)); // note: assumes utf8 terminal
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) start = Date.now();
  }

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  console.log(
    "\n\nachieved tok/s: %f\n",
    (pos - 1) / (Date.now() - start) * 1000.0,
  );
}

function error_usage(): never {
  console.error("Usage: ... llama2.ts <checkpoint> [options]");
  console.error('Example: llama2.ts model.bin -n 256 -i "Once upon a time"');
  console.error("Options:");
  console.error("  -t <float>  temperature, default 1.0");
  console.error(
    "  -p <float>  p value in top-p (nucleus) sampling. default 0.9, 0 = off",
  );
  console.error("  -s <int>    random seed, default time(NULL)");
  console.error(
    "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len",
  );
  console.error("  -i <string> input prompt");
  Deno.exit(1);
}

main();
