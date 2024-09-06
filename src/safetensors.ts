import * as fs from "node:fs";
import * as path from "node:path";
import { Buffer } from "node:buffer";
import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

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
  Q8_0: (buffer: Uint8Array, entry: HeaderEntry) => {
    const gs = entry.group_size ?? raise("No group size");
    const groups = buffer.byteLength /
      (gs + Float32Array.BYTES_PER_ELEMENT);
    const qlength = groups * gs;
    const q = new Int8Array(buffer.buffer, 0, qlength);
    const s = new Float32Array(buffer.buffer, qlength);

    return { q, s };
  },
};

type SUPPORTED_DTYPE_NAMES = keyof typeof DTYPE_MAPPING;
export type SUPPORTED_DTYPES = ReturnType<
  typeof DTYPE_MAPPING[SUPPORTED_DTYPE_NAMES]
>;

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
    | "Q8_0";
  shape: number[];
  offsets?: [number, number];
  data_offsets?: [number, number];
  group_size?: number;
};
type Header = {
  [key: string]: HeaderEntry;
} & {
  __metadata__: { [key: string]: string | number } & {
    format: "pt" | string;
    headerLength: number;
  };
};

export type WeightsEntry<D extends SUPPORTED_DTYPE_NAMES> = HeaderEntry & {
  weights: ReturnType<typeof DTYPE_MAPPING[D]>;
};
interface RecursiveWeights {
  [key: string]: Weights;
}
type Weights =
  & RecursiveWeights
  & { weight: WeightsEntry<SUPPORTED_DTYPE_NAMES> }
  & { layers: RecursiveWeights[] };

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
  return intype(dataRaw, entry);
}

type Safetensor = {
  metadata: Record<string, unknown>;
  weights: RecursiveWeights;
};

function readSafetensors(
  path: string,
  options: { filter?: (layer: string) => boolean } = {},
): Safetensor {
  const handle = fs.openSync(path, "r") as FileHandle;
  const header = readSTHeader(handle);

  const weights: RecursiveWeights = {};

  for (const [key, value] of Object.entries(header)) {
    if (key === "__metadata__") {
      continue;
    }
    if (options.filter && !options.filter(key)) {
      continue;
    }
    console.log("reading", key, value.shape);
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

  fs.closeSync(handle);

  return {
    metadata: header.__metadata__,
    weights,
  };
}

function readSafetensorsIndex(
  indexFile: string,
  options: { filter?: (layer: string) => boolean } = {},
): Safetensor {
  const index: {
    metadata: Record<string, unknown>;
    weight_map: Record<string, string>;
  } = JSON.parse(fs.readFileSync(indexFile, { encoding: "utf-8" }));
  const files = Object.values(index.weight_map).filter((f, i, a) =>
    a.indexOf(f) === i
  );

  let metadata = index.metadata;
  const weights: RecursiveWeights = {};
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
      console.log("reading", key, value.shape);
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
  }
  return {
    metadata: metadata,
    weights,
  };
}

export type TransformersModel = {
  config: Record<string, unknown>;
  metadata: Record<string, string>;
  weights: RecursiveWeights;
};

export function readHFRepo(
  configPath: string,
  modelPath: string,
  options: { filter?: (layer: string) => boolean } = {},
): TransformersModel {
  const config = JSON.parse(fs.readFileSync(configPath, { encoding: "utf-8" }));
  const indexFile = modelPath.endsWith(".index.json")
    ? modelPath
    : fs.existsSync(modelPath + ".index.json")
    ? modelPath + ".index.json"
    : "";
  if (indexFile) {
    const model = readSafetensorsIndex(indexFile, options);
    return { config, ...model } as TransformersModel;
  }
  const model = readSafetensors(modelPath, options);
  return { config, ...model } as TransformersModel;
}

function flattenWeightsToSafetensors(
  weights: RecursiveWeights,
  flattenedWeights: {
    [key: string]: WeightsEntry<SUPPORTED_DTYPE_NAMES>;
  } = {},
  prefix = "",
): { [key: string]: WeightsEntry<SUPPORTED_DTYPE_NAMES> } {
  for (const [key, value] of Object.entries(weights)) {
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) {
        flattenWeightsToSafetensors(
          value[i] as RecursiveWeights,
          flattenedWeights,
          `${prefix}${key}.${i}.`,
        );
      }
    } else {
      if (key === "weight") {
        flattenedWeights[`${prefix}${key}`] = value as any;
      } else if (typeof value === "object") {
        flattenWeightsToSafetensors(
          value as RecursiveWeights,
          flattenedWeights,
          `${prefix}${key}.`,
        );
      }
    }
  }
  return flattenedWeights;
}

Deno.test("flattenWeightsToSafetensors", () => {
  const weights = {
    model: {
      layers: [
        {
          test: {
            weight: {
              dtype: "F32",
              shape: [2, 2],
              weights: new Float32Array([1, 2, 3, 4]),
              data_offsets: [0, 16],
            },
          },
          test2: {
            weight: {
              dtype: "Q8_0",
              shape: [2, 2],
              group_size: 2,
              weights: {
                q: new Int8Array([1, 2, 3, 4]),
                s: new Float32Array([1, 2]),
              },
              data_offsets: [0, 16],
            },
          },
        },
      ],
    },
  };
  const flattenedWeights = flattenWeightsToSafetensors(weights as any);
  assertEquals(flattenedWeights, {
    "model.layers.0.test.weight": {
      dtype: "F32",
      shape: [2, 2],
      weights: new Float32Array([1, 2, 3, 4]),
      data_offsets: [0, 16],
    },
    "model.layers.0.test2.weight": {
      dtype: "Q8_0",
      shape: [2, 2],
      group_size: 2,
      weights: {
        q: new Int8Array([1, 2, 3, 4]),
        s: new Float32Array([1, 2]),
      },
      data_offsets: [0, 16],
    },
  });
});

export async function writeSafetensors(
  model: TransformersModel,
  configPath: string,
  modelPath: string,
) {
  const flattenedWeights = flattenWeightsToSafetensors(model.weights);
  let dataoffset = 0;
  const layerHeader = {
    "__metadata__": model.metadata,
    ...Object.fromEntries(
      Object.entries(flattenedWeights).map(([key, value]) => {
        const { dtype, shape, weights } = value;
        const start = dataoffset;
        let group_size = undefined;
        if ("q" in weights) {
          dataoffset = start + weights.q.byteLength + weights.s.byteLength;
          group_size = weights.q.length / weights.s.length;
        } else {
          dataoffset = start + weights.byteLength;
        }

        return [key, {
          dtype,
          shape,
          data_offsets: [start, dataoffset],
          group_size,
        }];
      }),
    ),
  };

  const header = new TextEncoder().encode(JSON.stringify(layerHeader));
  const headerSize = new Uint8Array(8);
  const headerView = new DataView(headerSize.buffer);
  headerView.setBigUint64(0, BigInt(header.byteLength), true);

  const filecontents: Uint8Array[] = [
    headerSize,
    header,
    ...Object.values(flattenedWeights).map((w) => {
      if (
        "q" in w.weights && "s" in w.weights
      ) {
        const q = w.weights.q;
        const s = w.weights.s;
        const buffer = new Uint8Array(
          w.weights.q.byteLength + w.weights.s.byteLength,
        );
        buffer.set(q);
        buffer.set(
          new Uint8Array(s.buffer, s.byteOffset, s.byteLength),
          q.byteLength,
        );
        return buffer;
      }

      const buffer = new Uint8Array(w.weights.byteLength);
      buffer.set(
        new Uint8Array(
          w.weights.buffer,
          w.weights.byteOffset,
          w.weights.byteLength,
        ),
      );
      return buffer;
    }),
  ];

  const configFileHandle = await Deno.open(configPath, {
    create: true,
    write: true,
  });
  const modelFileHandle = await Deno.open(modelPath, {
    create: true,
    write: true,
  });
  await configFileHandle.write(
    new TextEncoder().encode(JSON.stringify(model.config)),
  );
  for (const filecontent of filecontents) {
    let written = 0;
    while (written < filecontent.byteLength) {
      written += await modelFileHandle.write(filecontent.subarray(written));
    }
  }
  configFileHandle.close();
  modelFileHandle.close();
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

export function dequantize(
  o: Float32Array,
  x: { q: Int8Array; s: Float32Array },
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    o[i] = x.q[i] * x.s[Math.floor(i / (x.q.length / x.s.length))];
  }
}
