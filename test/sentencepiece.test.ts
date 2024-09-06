import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { encode, readHFRepoTokenizer } from "../src/sentencepiece.ts";

/*

// test 1
char *prompt = "I believe the meaning of life is";
int expected_tokens[] = {1, 306, 4658, 278, 6593, 310, 2834, 338};
test_prompt_encoding(&tokenizer, prompt, expected_tokens, sizeof(expected_tokens) / sizeof(int));

// test 2
char* prompt2 = "Simply put, the theory of relativity states that ";
int expected_tokens2[] = {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871};
test_prompt_encoding(&tokenizer, prompt2, expected_tokens2, sizeof(expected_tokens2) / sizeof(int));

// test 3
char* prompt3 = "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ";
int expected_tokens3[] = {1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871};
test_prompt_encoding(&tokenizer, prompt3, expected_tokens3, sizeof(expected_tokens3) / sizeof(int));

// test 4
char* prompt4 = "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>";
int expected_tokens4[] = {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149};
test_prompt_encoding(&tokenizer, prompt4, expected_tokens4, sizeof(expected_tokens4) / sizeof(int));

*/
Deno.test("encode test case 1", () => {
  const hf = readHFRepoTokenizer(
    "../llama2.c/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json",
    "../llama2.c/TinyLlama-1.1B-Chat-v1.0/tokenizer.json",
  );

  const prompt = "I believe the meaning of life is";
  const expected_tokens = [1, 306, 4658, 278, 6593, 310, 2834, 338];
  const tokens = encode(prompt, hf);
  assertEquals(tokens, expected_tokens);
});

Deno.test("encode test case 2", () => {
  const hf = readHFRepoTokenizer(
    "../llama2.c/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json",
    "../llama2.c/TinyLlama-1.1B-Chat-v1.0/tokenizer.json",
  );

  const prompt = "Simply put, the theory of relativity states that ";
  const expected_tokens = [
    1,
    3439,
    17632,
    1925,
    29892,
    278,
    6368,
    310,
    14215,
    537,
    5922,
    393,
    29871,
  ];
  const tokens = encode(prompt, hf);
  assertEquals(tokens, expected_tokens);
});

Deno.test("encode test case 3", () => {
  const hf = readHFRepoTokenizer(
    "../llama2.c/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json",
    "../llama2.c/TinyLlama-1.1B-Chat-v1.0/tokenizer.json",
  );

  const prompt =
    "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ";
  const expected_tokens = [
    1,
    319,
    11473,
    2643,
    378,
    629,
    271,
    18099,
    278,
    3815,
    373,
    278,
    6826,
    29901,
    13,
    13,
    4706,
    6324,
    14332,
    29892,
    13,
    13,
    4706,
    306,
    925,
    29871,
  ];
  const tokens = encode(prompt, hf);
  assertEquals(tokens, expected_tokens);
});
