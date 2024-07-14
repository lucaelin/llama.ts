export function encode(
  text: string,
  bos: boolean,
  eos: boolean,
  vocab: string[],
  vocab_scores: number[],
  tokens: Int32Array,
) {
  // first encode every individual byte in the input string
  let n_tokens = 0; // the number of tokens
  if (bos) tokens[n_tokens++] = 1; // BOS token
  for (let i = 0; i < text.length; ++i) {
    const id = vocab.indexOf(text.charAt(i));
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
      const str_buffer = vocab[tokens[i]] + vocab[tokens[i + 1]];
      const id = vocab.indexOf(str_buffer);
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

export function decode(
  vocab: string[],
  prev_token: number,
  token: number,
): string {
  //console.log("token: %d %d", prev_token, token);
  let piece = vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece.charAt(0) == " ") piece = piece.substring(1);

  return piece;
}
