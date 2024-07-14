import { argmax, softmax } from "./kernels.ts";
import { random_f32 } from "./rng.ts";

type float = number;

export function sample(
  logits: Float32Array,
  vocab_size: number,
  temperature: number,
  topp: number,
) {
  // sample the token given the logits and some hyperparameters
  if (temperature == 0.0) {
    // greedy argmax sampling: take the token with the highest probability
    return argmax(logits, vocab_size);
  } else {
    // apply the temperature to the logits
    for (let q = 0; q < vocab_size; q++) {
      logits[q] /= temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, vocab_size);
    const coin = random_f32();
    // we sample from this distribution to get the next token
    if (topp <= 0 || topp >= 1) {
      // simply sample from the predicted probability distribution
      return sample_mult(logits, vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      return sample_topp(
        logits,
        vocab_size,
        topp,
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
  vocab_size: float,
  topp: number,
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

  const probindex: { prob: float; index: number }[] = new Array(vocab_size);
  const cutoff = (1.0 - topp) / (vocab_size - 1);
  for (let i = 0; i < vocab_size; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0++] = { index: i, prob: probabilities[i] };
    }
  }
  probindex.sort((a, b) => b.prob - a.prob);

  // truncate the list where cumulative probability exceeds topp
  let cumulativeProb = 0;
  let lastIdx = n0 - 1; // in case of rounding errors consider all elements
  for (let i = 0; i < vocab_size; i++) {
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
