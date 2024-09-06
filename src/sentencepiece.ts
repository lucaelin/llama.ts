import fs from "node:fs";

export function encode(
  text: string,
  tokenizer: HFRepoTokenizer,
) {
  const vocab = tokenizer.tokenizer.model.vocab;
  const merges = tokenizer.tokenizer.model.merges.map((m) =>
    m.split(" ") as [string, string]
  );
  const bos_token = typeof tokenizer.config.bos_token === "object"
    ? tokenizer.config.bos_token.content
    : tokenizer.config.bos_token;
  const bos_token_id = vocab[
    bos_token
  ] ?? tokenizer.tokenizer.added_tokens?.find((t) =>
    t.content === bos_token
  )?.id;

  const eos_token = typeof tokenizer.config.eos_token === "object"
    ? tokenizer.config.eos_token.content
    : tokenizer.config.eos_token;
  const eos_token_id = vocab[
    eos_token
  ] ?? tokenizer.tokenizer.added_tokens?.find((t) =>
    t.content === eos_token
  )?.id;

  if (!text) {
    return tokenizer.config.add_bos_token ?? true ? [bos_token_id] : [];
  }

  const prepend = tokenizer.tokenizer.normalizer?.type === "Sequence"
    ? tokenizer.tokenizer.normalizer?.normalizers.find(
      (n) => n.type === "Prepend",
    )?.prepend ?? ""
    : tokenizer.tokenizer.normalizer?.type === "Prepend"
    ? tokenizer.tokenizer.normalizer?.prepend ?? ""
    : "";
  const replaceSpace = tokenizer.tokenizer.normalizer?.type === "Sequence"
    ? (tokenizer.tokenizer.normalizer?.normalizers.find(
      (n) => n.type === "Replace" && n.pattern.String === " ",
    ) as any)?.content ?? "▁"
    : tokenizer.tokenizer.normalizer?.type === "Replace"
    ? tokenizer.tokenizer.normalizer.content ?? "▁"
    : "▁";

  const tokens = (prepend + text).replaceAll(" ", replaceSpace).split("");

  for (const added_token of tokenizer.tokenizer.added_tokens ?? []) {
    if (!added_token.special) continue;
    for (let i = 0; i < tokens.length; i++) {
      const slice = tokens.slice(i, i + added_token.content.length);
      if (slice.join("") === added_token.content) {
        tokens.splice(i, added_token.content.length, added_token.content);
      }
    }
  }

  while (true) {
    let candidate = Infinity;
    for (let i = 0; i < tokens.length - 1; i++) {
      const token = tokens[i];
      const next_token = tokens[i + 1];
      const newCandidate = merges.findIndex(([a, b]) =>
        a == token && b == next_token
      );
      if (newCandidate >= 0 && candidate > newCandidate) {
        candidate = newCandidate;
      }
    }
    if (candidate === Infinity) break;
    const [a, b] = merges[candidate];
    const pos = tokens.findIndex((t, i, tokens) =>
      tokens[i] === a && tokens[i + 1] === b
    );
    tokens.splice(pos, 2, a + b);
  }

  const token_ids = tokens.map((token) => {
    const byteToken = "<0x" +
      "\n".charCodeAt(0).toString(16).toUpperCase().padStart(2, "0") + ">";
    return vocab[token] ?? vocab[byteToken] ?? vocab["<unk>"];
  });

  if (tokenizer.config.add_bos_token ?? true) token_ids.unshift(bos_token_id);
  if (tokenizer.config.add_eos_token ?? false) token_ids.push(eos_token_id);

  return token_ids;
}

export function decode(
  prev_token: number,
  token: number,
  tokenizer: HFRepoTokenizer,
): string {
  const vocab = Object.fromEntries(
    Object.entries(tokenizer.tokenizer.model.vocab).map(([a, b]) => [b, a]),
  );

  //console.log("token: %d %d", prev_token, token);
  let piece = vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece.charAt(0) === "▁") piece = piece.substring(1);
  if (piece.startsWith("<0x") && piece.endsWith(">")) {
    const byte = parseInt(piece.substring(4, 6), 16);
    piece = String.fromCharCode(byte);
  }

  return piece.replaceAll("▁", " ");
}

type HFNormalizer =
  | {
    "type": "Prepend";
    "prepend": string;
  }
  | {
    "type": "Replace";
    "pattern": {
      "String": string;
    };
    "content": string;
  }
  | {
    "type": "Sequence";
    "normalizers": HFNormalizer[];
  };

type HFRepoTokenizer = {
  config: {
    add_bos_token?: boolean;
    add_eos_token?: boolean;
    bos_token: { content: string } | string;
    eos_token: { content: string } | string;
  };
  tokenizer: {
    model: { vocab: { [key: string]: number }; merges: string[] };
    added_tokens?: {
      "id": number;
      "content": string;
      "single_word": boolean;
      "lstrip": boolean;
      "rstrip": boolean;
      "normalized": boolean;
      "special": boolean;
    }[];

    normalizer?: HFNormalizer;
  };
};

export function readHFRepoTokenizer(
  configPath: string,
  tokenizerPath: string,
  options: {} = {},
): HFRepoTokenizer {
  const config = JSON.parse(fs.readFileSync(configPath, { encoding: "utf-8" }));
  const tokenizer = JSON.parse(
    fs.readFileSync(tokenizerPath, { encoding: "utf-8" }),
  );
  return { config, tokenizer };
}
