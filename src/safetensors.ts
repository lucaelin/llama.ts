import * as fs from "node:fs";
import * as path from "node:path";
import { Buffer } from "node:buffer";
import {
  DEFAULT_GROUP_SIZE,
  dequantize,
  newQ8ArrayFrom,
  Q8Array,
  q8ArrayToFloat32Array,
} from "./quantization.ts";

function raise(message: string): never {
  throw new Error(message);
}

const DTYPE_MAPPING = {
  F64: (buffer: Uint8Array) =>
    new Float64Array(
      buffer.buffer,
      0,
      buffer.byteLength / Float64Array.BYTES_PER_ELEMENT,
    ),
  F32: (buffer: Uint8Array) =>
    new Float32Array(
      buffer.buffer,
      0,
      buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
    ),
  F16: (buffer: Uint8Array) =>
    new Float16Array(
      buffer.buffer,
      0,
      buffer.byteLength / Float16Array.BYTES_PER_ELEMENT,
    ),
  BF16: (buffer: Uint8Array) => bf16ToF32(buffer),
  I64: (buffer: Uint8Array) =>
    new BigInt64Array(
      buffer.buffer,
      0,
      buffer.byteLength / BigInt64Array.BYTES_PER_ELEMENT,
    ),
  I32: (buffer: Uint8Array) =>
    new Int32Array(
      buffer.buffer,
      0,
      buffer.byteLength / Int32Array.BYTES_PER_ELEMENT,
    ),
  I16: (buffer: Uint8Array) =>
    new Int16Array(
      buffer.buffer,
      0,
      buffer.byteLength / Int16Array.BYTES_PER_ELEMENT,
    ),
  I8: (buffer: Uint8Array) =>
    new Int8Array(
      buffer.buffer,
      0,
      buffer.byteLength / Int8Array.BYTES_PER_ELEMENT,
    ),
  U8: (buffer: Uint8Array) =>
    new Uint8Array(
      buffer.buffer,
      0,
      buffer.byteLength / Uint8Array.BYTES_PER_ELEMENT,
    ),
  BOOL: (buffer: Uint8Array, offset?: number, length?: number) =>
    new Uint8Array(buffer.buffer, offset, length),
};

const OUTPUT_DTYPE_MAPPING = {
  "F32": (buffer: ReturnType<typeof DTYPE_MAPPING[SUPPORTED_DTYPE_NAMES]>) =>
    new Float32Array(buffer),
  "Q8_0": (buffer: ReturnType<typeof DTYPE_MAPPING[SUPPORTED_DTYPE_NAMES]>) =>
    newQ8ArrayFrom(
      buffer instanceof Float32Array ? buffer : new Float32Array(buffer),
      DEFAULT_GROUP_SIZE,
    ),
  "raw": (buffer: ReturnType<typeof DTYPE_MAPPING[SUPPORTED_DTYPE_NAMES]>) =>
    new Uint8Array(buffer.buffer, 0, buffer.byteLength),
  "unknown": (
    buffer: ReturnType<typeof DTYPE_MAPPING[SUPPORTED_DTYPE_NAMES]>,
  ) => buffer,
};

type SUPPORTED_DTYPE_NAMES = keyof typeof DTYPE_MAPPING;
type SUPPORTED_DTYPES = ReturnType<typeof DTYPE_MAPPING[SUPPORTED_DTYPE_NAMES]>;

type OUTPUT_DTYPE_NAMES = keyof typeof OUTPUT_DTYPE_MAPPING;

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
  __metadata__: { [key: string]: string | number } & {
    format: "pt" | string;
    headerLength: number;
  };
};

type WeightsEntry<D extends OUTPUT_DTYPE_NAMES> = HeaderEntry & {
  weights: ReturnType<typeof OUTPUT_DTYPE_MAPPING[D]>;
};
interface RecursiveWeights<D extends OUTPUT_DTYPE_NAMES> {
  [key: string]: Weights<D>;
}
type Weights<D extends OUTPUT_DTYPE_NAMES> =
  & RecursiveWeights<D>
  & { weight: WeightsEntry<D> }
  & { layers: RecursiveWeights<D>[] };

function readSTHeader(handle: FileHandle): Header {
  const prefixBytes = new Uint8Array(8);
  fs.readSync(handle, prefixBytes, 0, prefixBytes.length, 0);
  const actualHeaderLength = new DataView(prefixBytes.buffer)
    .getBigUint64(0, true);
  const headerLength = new DataView(prefixBytes.buffer)
    .getUint32(0, true);
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
): ReturnType<typeof DTYPE_MAPPING[SUPPORTED_DTYPE_NAMES]> {
  const intype = DTYPE_MAPPING[entry.dtype];
  const [start, end] = entry.offsets ?? entry.data_offsets ??
    raise("No offsets");
  const length = end - start;
  const dataRaw = Buffer.alloc(length);
  fs.readSync(handle, dataRaw, 0, length, start + headerSize);
  if (entry.dtype === "BF16") {
    return bf16ToF32(new Uint8Array(dataRaw));
  }
  return intype(dataRaw);
}

type Safetensor<D extends OUTPUT_DTYPE_NAMES> = {
  metadata: Record<string, unknown>;
  weights: RecursiveWeights<D>;
  dtype: D;
};

function readSafetensors<const D extends OUTPUT_DTYPE_NAMES>(
  path: string,
  dtype: D,
  options: { filter?: (layer: string) => boolean } = {},
): Safetensor<D> {
  const handle = fs.openSync(path, "r") as FileHandle;
  const header = readSTHeader(handle);

  const weights: RecursiveWeights<D> = {};

  for (const [key, value] of Object.entries(header)) {
    if (key === "__metadata__") {
      continue;
    }
    if (options.filter && !options.filter(key)) {
      continue;
    }
    console.log("reading", key, value.shape, dtype);
    const dataRaw = readSTWeights(
      handle,
      header.__metadata__.headerLength,
      value as HeaderEntry,
    );
    const data = OUTPUT_DTYPE_MAPPING[dtype](dataRaw);

    const path = key.split(".");
    let current: Weights<D> = weights as Weights<D>;
    for (let i = 0; i < path.length - 1; i++) {
      if (!current[path[i]]) {
        current[path[i]] = path[i] === "layers" ? ([] as any) : {};
      }
      current = current[path[i]] as Weights<D>;
    }
    current[path[path.length - 1]] = {
      ...(value as HeaderEntry),
      weights: data,
    } as any;
  }

  fs.closeSync(handle);

  return {
    metadata: header.__metadata__,
    weights,
    dtype,
  };
}

function readSafetensorsIndex<const D extends OUTPUT_DTYPE_NAMES>(
  indexFile: string,
  dtype: D,
  options: { filter?: (layer: string) => boolean } = {},
): Safetensor<D> {
  const index: {
    metadata: Record<string, unknown>;
    weight_map: Record<string, string>;
  } = JSON.parse(fs.readFileSync(indexFile, { encoding: "utf-8" }));
  const files = Object.values(index.weight_map).filter((f, i, a) =>
    a.indexOf(f) === i
  );

  let metadata = index.metadata;
  const weights: RecursiveWeights<D> = {};
  for (const file of files) {
    const filePath = path.dirname(indexFile) + "/" + file;

    const handle = fs.openSync(filePath, "r") as FileHandle;
    const header = readSTHeader(handle);

    for (const [key, value] of Object.entries(header)) {
      if (key === "__metadata__") {
        metadata = { ...metadata, ...value };
        continue;
      }
      if (options.filter && !options.filter(key)) {
        continue;
      }
      console.log("reading", key, value.shape, dtype);
      const dataRaw = readSTWeights(
        handle,
        header.__metadata__.headerLength,
        value as HeaderEntry,
      );
      const data = OUTPUT_DTYPE_MAPPING[dtype](dataRaw);

      const path = key.split(".");
      let current: Weights<D> = weights as Weights<D>;
      for (let i = 0; i < path.length - 1; i++) {
        if (!current[path[i]]) {
          current[path[i]] = path[i] === "layers" ? ([] as any) : {};
        }
        current = current[path[i]] as Weights<D>;
      }
      current[path[path.length - 1]] = {
        ...(value as HeaderEntry),
        weights: data,
      } as any;
    }
  }
  return {
    metadata: metadata,
    weights,
    dtype,
  };
}

export type TransformersModel<D extends OUTPUT_DTYPE_NAMES> = {
  config: Record<string, unknown>;
  metadata: Record<string, string>;
  weights: RecursiveWeights<D>;
  dtype: D;
};

export function readHFRepo<const D extends OUTPUT_DTYPE_NAMES>(
  configPath: string,
  modelPath: string,
  dtype: D,
  options: { filter?: (layer: string) => boolean } = {},
): TransformersModel<D> {
  const config = JSON.parse(fs.readFileSync(configPath, { encoding: "utf-8" }));
  const indexFile = modelPath.endsWith(".index.json")
    ? modelPath
    : fs.existsSync(modelPath + ".index.json")
    ? modelPath + ".index.json"
    : "";
  if (indexFile) {
    const model = readSafetensorsIndex(indexFile, dtype, options);
    if (dtype === "Q8_0") model.metadata.group_size = DEFAULT_GROUP_SIZE;
    return { config, ...model } as TransformersModel<D>;
  }
  const model = readSafetensors(modelPath, dtype, options);
  if (dtype === "Q8_0") model.metadata.group_size = DEFAULT_GROUP_SIZE;
  return { config, ...model } as TransformersModel<D>;
}

export function fp16ToF32(input: Uint8Array) {
  const input16 = new Float16Array(input.buffer);
  const output = new Float32Array(input16.length);
  for (let i = 0; i < input16.length; i++) {
    output[i] = input16[i];
  }
  return output;
}

export function bf16ToF32(input: Uint8Array) {
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

export function nf4tof32(input: Uint8Array) {
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

export function permuteReverse<W extends Float32Array | Q8Array>(
  win: W,
  n_heads: number,
  dim1: number,
  dim2: number,
): W {
  const isQ8 = win instanceof Q8Array;
  const w = isQ8
    ? q8ArrayToFloat32Array(win, win.q.length / win.s.length) as Float32Array
    : win as Float32Array;

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

  if (isQ8) {
    return newQ8ArrayFrom(result, win.gs) as W;
  }
  return result as W;
}
