import * as fs from "node:fs";
import { Buffer } from "node:buffer";

function raise(message: string): never {
  throw new Error(message);
}

type FileHandle = number & { [Symbol.toStringTag]: "FileHandle" };

type HeaderEntry = {
  dtype:
    | "F64"
    | "F32"
    | "F16"
    | "BF16"
    | "I64"
    | "I32"
    | "I16"
    | "I8"
    | "U8"
    | "BOOL";
  shape: number[];
  offsets?: [number, number];
  data_offsets?: [number, number];
};
type Header = {
  [key: string]: HeaderEntry;
} & {
  __metadata__: { [key: string]: string } & {
    format: "pt" | string;
    headerLength: number;
  };
};

type WeightsEntry = HeaderEntry & { weights: ArrayBuffer };
interface RecursiveWeights {
  [key: string]: Weights;
}
type Weights =
  & RecursiveWeights
  & { weight: WeightsEntry }
  & { layers: RecursiveWeights[] };

function readSTHeader(handle: FileHandle): Header {
  const prefixBytes = new Uint8Array(8);
  fs.readSync(handle, prefixBytes, 0, prefixBytes.length, 0);
  const actualHeaderLength = new DataView(prefixBytes.buffer).getBigUint64(
    0,
    true,
  );
  const headerLength = new DataView(prefixBytes.buffer).getUint32(0, true);
  if (actualHeaderLength > headerLength) {
    throw new Error("Header too large");
  }
  const headerBytes = new Uint8Array(headerLength);
  fs.readSync(handle, headerBytes, 0, headerLength, 8);
  const header = new TextDecoder().decode(headerBytes);
  const json = JSON.parse(header);
  const bytelength = headerLength + 8;
  return {
    ...json,
    __metadata__: { ...json.__metadata__, headerLength: bytelength },
  };
}

function readSTWeights(
  handle: FileHandle,
  headerSize: number,
  entry: HeaderEntry,
): ArrayBuffer {
  const dtype = {
    F64: Float64Array,
    F32: Float32Array,
    F16: Float16Array,
    BF16: Float32Array,
    I64: BigInt64Array,
    I32: Int32Array,
    I16: Int16Array,
    I8: Int8Array,
    U8: Uint8Array,
    BOOL: Uint8Array,
  }[entry.dtype];
  const [start, end] = entry.offsets ?? entry.data_offsets ??
    raise("No offsets");
  const length = end - start;
  const data = Buffer.alloc(length);
  fs.readSync(handle, data, 0, length, start + headerSize);
  if (entry.dtype === "BF16") {
    const input = new Uint8Array(data);
    return bf16ToF32(input);
  }
  if (entry.dtype === "F16") {
    const input = new Uint8Array(data);
    return fp16ToF32(input);
  }
  return new dtype(
    data.buffer,
    data.byteOffset,
    data.length / dtype.BYTES_PER_ELEMENT,
  );
}

function readSafetensors(path: string) {
  const handle = fs.openSync(path, "r") as FileHandle;
  const header = readSTHeader(handle);

  const weights: RecursiveWeights = {};

  for (const [key, value] of Object.entries(header)) {
    // console.log("reading", key, value.shape);
    if (key === "__metadata__") {
      continue;
    }
    const data = readSTWeights(
      handle,
      header.__metadata__.headerLength,
      value as HeaderEntry,
    );
    const path = key.split(".");
    let current: Weights = weights as Weights;
    for (let i = 0; i < path.length - 1; i++) {
      if (!current[path[i]]) {
        current[path[i]] = path[i] === "layers" ? ([] as any) : {};
      }
      current = current[path[i]] as Weights;
    }
    current[path[path.length - 1]] = {
      ...(value as HeaderEntry),
      weights: data,
    } as any;
  }

  return {
    metadata: header.__metadata__,
    weights,
  };
}

export type TransformersModel = {
  config: Record<string, any>;
  metadata: Record<string, string>;
  weights: RecursiveWeights;
};

export function readHFRepo(configPath: string, modelPath: string) {
  const config = JSON.parse(fs.readFileSync(configPath, { encoding: "utf-8" }));
  const model = readSafetensors(modelPath) as {
    metadata: Record<string, string>;
    weights: RecursiveWeights;
  };
  return { config, ...model } as TransformersModel;
}

/*
// 1 and -2
console.log(
  bf16ToF32(new Uint8Array([0b10000000, 0b00111111, 0b00000000, 0b11000000])),
);

// 0 and -0
console.log(
  bf16ToF32(new Uint8Array([0b00000000, 0b00000000, 0b00000000, 0b10000000])),
);

// inf and -inf
console.log(
  bf16ToF32(new Uint8Array([0b10000000, 0b01111111, 0b10000000, 0b11111111])),
);

// qnan and snan
console.log(
  bf16ToF32(new Uint8Array([0b11000001, 0b01111111, 0b10000001, 0b01111111])),
);

// pi and 1/3
console.log(
  bf16ToF32(new Uint8Array([0b01001001, 0b01000000, 0b10101011, 0b00111110])),
);

console.log(
  bf16ToF32(new Uint8Array([0xcd, 0x3d, 0x24, 0x3c, 0x83, 0x3a])),
);

const { metadata, weights } = readSafetensors(
  "../llama2.c/TinyLlama-1.1B-Medical/adapter_model.safetensors",
);
console.log(
  weights.base_model.model.model.layers[15].self_attn.v_proj.lora_A.weight
    .weights,
);
*/

function fp16ToF32(input: Uint8Array) {
  const input16 = new Float16Array(input.buffer);
  const output = new Float32Array(input16.length);
  for (let i = 0; i < input16.length; i++) {
    output[i] = input16[i];
  }
  return output;
}

function bf16ToF32(input: Uint8Array) {
  const output = new Float32Array(input.length / 2);
  for (let i = 0; i < output.length; i++) {
    const msb = input[i * 2 + 0];
    const lsb = input[i * 2 + 1];
    const sign = (lsb & 0b10000000) >> 7;
    const exponent = ((lsb & 0b01111111) << 1) +
      ((msb & 0b10000000) >> 7);
    const mantissa = msb & 0b01111111;

    if (exponent === 0 && mantissa === 0) {
      output[i] = sign ? -0 : 0;
      continue;
    }
    if (exponent === 0b11111111 && mantissa === 0) {
      output[i] = sign ? -Infinity : Infinity;
      continue;
    }
    if (
      exponent === 0b11111111
    ) {
      output[i] = NaN;
      continue;
    }

    output[i] = Math.pow(-1, sign) *
      Math.pow(2, exponent - 127) * (1 + mantissa / 128);
  }
  return output;
}

function _nf4tof32(input: Uint8Array) {
  const output = new Float32Array(input.length * 2);
  for (let i = 0; i < output.length; i += 2) {
    const val1 = input[i] & 0b00001111;
    const val2 = input[i] & 0b11110000 >> 4;

    output[i] = nf4tof32Single(val1);
    output[i + 1] = nf4tof32Single(val2);
  }

  return output;
}

function nf4tof32Single(val: number) {
  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py of bitsandbytes
  if ((val & 0b1000) == 8) {
    if ((val & 0b0100) == 4) { // 1
      if ((val & 0b0010) == 2) { // 11
        if ((val & 0b0001) == 1) { // 111
          return 1.0;
        } else {
          return 0.7229568362236023;
        }
      } else if ((val & 0b0001) == 1) { // 110
        return 0.5626170039176941;
      } else {
        return 0.44070982933044434;
      }
    } else if ((val & 0b0010) == 2) { //10
      if ((val & 0b0001) == 1) { // 101
        return 0.33791524171829224;
      } else {
        return 0.24611230194568634;
      }
    } else if ((val & 0b0001) == 1) { // 100
      return 0.16093020141124725;
    } else {
      return 0.07958029955625534;
    }
  } else if ((val & 0b0100) == 4) { // 0
    if ((val & 0b0010) == 2) { //01
      if ((val & 0b0001) == 1) { // 011
        return 0.0;
      } else {
        return -0.09105003625154495;
      }
    } else if ((val & 0b0001) == 1) { // 010
      return -0.18477343022823334;
    } else {
      return -0.28444138169288635;
    }
  } else if ((val & 0b0010) == 2) { //00
    if ((val & 0b0001) == 1) { // 001
      return -0.39491748809814453;
    } else {
      return -0.5250730514526367;
    }
  } else if ((val & 0b0001) == 1) { // 000
    return -0.6961928009986877;
  } else {
    return -1.0;
  }
}

export function permuteReverse(
  w: Float32Array,
  n_heads: number,
  dim1: number,
  dim2: number,
): Float32Array {
  const newShape = [n_heads, 2, Math.floor(dim1 / n_heads / 2), dim2];

  // Reshape w into newShape
  const reshaped: number[][][][] = [];
  let index = 0;
  for (let i = 0; i < newShape[0]; i++) {
    reshaped[i] = [];
    for (let j = 0; j < newShape[1]; j++) {
      reshaped[i][j] = [];
      for (let k = 0; k < newShape[2]; k++) {
        reshaped[i][j][k] = [];
        for (let l = 0; l < newShape[3]; l++) {
          reshaped[i][j][k][l] = w[index++];
        }
      }
    }
  }

  // Transpose (1, 2) => (0, 2, 1, 3)
  const transposed: number[][][][] = [];
  for (let i = 0; i < newShape[0]; i++) {
    transposed[i] = [];
    for (let k = 0; k < newShape[2]; k++) {
      transposed[i][k] = [];
      for (let j = 0; j < newShape[1]; j++) {
        transposed[i][k][j] = reshaped[i][j][k];
      }
    }
  }

  // Flatten the transposed array and reshape it into [dim1, dim2]
  const flattened: number[] = [];
  for (let i = 0; i < newShape[0]; i++) {
    for (let k = 0; k < newShape[2]; k++) {
      for (let j = 0; j < newShape[1]; j++) {
        for (let l = 0; l < newShape[3]; l++) {
          flattened.push(transposed[i][k][j][l]);
        }
      }
    }
  }

  const result = new Float32Array(dim1 * dim2);
  for (let i = 0; i < result.length; i++) {
    result[i] = flattened[i];
  }

  return result;
}
