/**
 * Why do we do all this?
 * We want to be able to dynamically allocate memory in WebAssembly and have it be accessible from JavaScript.
 * The way we do this is by creating a WebAssembly.Memory object and then pointing to it from both JavaScript and WebAssembly.
 * One problem is that growing the memory can cause the memory to be reallocated, which would invalidate the pointers on the JavaScript side.
 * So instead we create the Javascript TypedArrays on the fly and point to the correct memory address consistently.
 */

declare global {
  interface ArrayBuffer {
    [Symbol.metadata]: { name: "ArrayBuffer" };
  }
}

import { dequantize, quantize as quantize_js } from "./kernels.ts";
import { allocate, memory, quantize } from "./kernels_wasm.ts";
import {
  F32Tensor,
  I8Tensor,
  JSF32Tensor,
  JSQ8Tensor,
  Q8Tensor,
  U8Tensor,
} from "./types.ts";
import { Tensor } from "./types.ts";

export const WASM_ADDRESS = Symbol("WASM_ADDRESS");

export abstract class WasmTensor extends Tensor {
  static allocate(_n: number): Tensor {
    throw new Error("Not implemented");
  }
  static allocateFrom(_x: number[]) {
    throw new Error("Not implemented");
  }
  abstract get array(): ArrayLike<number>;
  abstract subarray(start: number, end: number): Tensor;
  abstract length: number;
  abstract [WASM_ADDRESS]: number;
}

export class WasmF32Tensor extends WasmTensor implements F32Tensor {
  static allocate(length: number) {
    const bytelength = length * Float32Array.BYTES_PER_ELEMENT;
    const ptr = allocate(bytelength, memory);
    return new WasmF32Tensor(bytelength, ptr, memory);
  }
  static allocateFrom(data: number[]) {
    const tensor = WasmF32Tensor.allocate(data.length);
    tensor.array.set(data);
    return tensor;
  }
  static moveFrom(data: Float32Array) {
    const tensor = WasmF32Tensor.allocate(data.length);
    tensor.array.set(data);
    return tensor;
  }

  constructor(
    public bytelength: number,
    public wasmAddress: number,
    public wasmMemory: WebAssembly.Memory = memory,
  ) {
    super();
  }

  get [WASM_ADDRESS]() {
    return this.wasmAddress;
  }

  get array() {
    return new Float32Array(this.wasmMemory.buffer).subarray(
      this.wasmAddress / Float32Array.BYTES_PER_ELEMENT,
      (this.wasmAddress + this.bytelength) / Float32Array.BYTES_PER_ELEMENT,
    );
  }

  subarray(start: number, end: number): WasmF32Tensor {
    return new WasmF32Tensor(
      (end - start) * Float32Array.BYTES_PER_ELEMENT,
      this.wasmAddress + start * Float32Array.BYTES_PER_ELEMENT,
      this.wasmMemory,
    );
  }

  get length() {
    return this.bytelength / Float32Array.BYTES_PER_ELEMENT;
  }
}

export class WasmI8Tensor extends WasmTensor implements I8Tensor {
  static allocate(length: number) {
    const bytelength = length * Int8Array.BYTES_PER_ELEMENT;
    const ptr = allocate(bytelength, memory);
    return new WasmI8Tensor(bytelength, ptr, memory);
  }
  static allocateFrom(data: number[]) {
    const tensor = WasmI8Tensor.allocate(data.length);
    tensor.array.set(data);
    return tensor;
  }

  constructor(
    public bytelength: number,
    public wasmAddress: number,
    public wasmMemory: WebAssembly.Memory = memory,
  ) {
    super();
  }

  get [WASM_ADDRESS]() {
    return this.wasmAddress;
  }

  get array() {
    return new Int8Array(this.wasmMemory.buffer).subarray(
      this.wasmAddress / Int8Array.BYTES_PER_ELEMENT,
      (this.wasmAddress + this.bytelength) / Int8Array.BYTES_PER_ELEMENT,
    );
  }

  subarray(start: number, end: number): WasmI8Tensor {
    return new WasmI8Tensor(
      (end - start) * Int8Array.BYTES_PER_ELEMENT,
      this.wasmAddress + start * Int8Array.BYTES_PER_ELEMENT,
      this.wasmMemory,
    );
  }

  get length() {
    return this.bytelength / Int8Array.BYTES_PER_ELEMENT;
  }
}

export class WasmU8Tensor extends WasmTensor implements U8Tensor {
  static allocate(length: number) {
    const bytelength = length * Uint8Array.BYTES_PER_ELEMENT;
    const ptr = allocate(bytelength, memory);
    return new WasmU8Tensor(bytelength, ptr, memory);
  }
  static allocateFrom(data: number[]) {
    const tensor = WasmU8Tensor.allocate(data.length);
    tensor.array.set(data);
    return tensor;
  }

  constructor(
    public bytelength: number,
    public wasmAddress: number,
    public wasmMemory: WebAssembly.Memory = memory,
  ) {
    super();
  }

  get [WASM_ADDRESS]() {
    return this.wasmAddress;
  }

  get array() {
    return new Uint8Array(this.wasmMemory.buffer).subarray(
      this.wasmAddress / Uint8Array.BYTES_PER_ELEMENT,
      (this.wasmAddress + this.bytelength) / Uint8Array.BYTES_PER_ELEMENT,
    );
  }

  subarray(start: number, end: number): WasmU8Tensor {
    return new WasmU8Tensor(
      (end - start) * Uint8Array.BYTES_PER_ELEMENT,
      this.wasmAddress + start * Uint8Array.BYTES_PER_ELEMENT,
      this.wasmMemory,
    );
  }

  get length() {
    return this.bytelength / Uint8Array.BYTES_PER_ELEMENT;
  }
}

export class WasmQ8Tensor extends WasmTensor implements Q8Tensor {
  static allocate(
    length: number,
    gs: number = JSQ8Tensor.DEFAULT_GROUP_SIZE,
  ) {
    const q = WasmI8Tensor.allocate(length);
    const s = WasmF32Tensor.allocate(length / gs);
    return new WasmQ8Tensor(q, s);
  }
  static allocateFrom(data: number[]) {
    const tensor = WasmQ8Tensor.allocate(data.length);
    tensor.array.set(data);
    return tensor;
  }

  static allocateFromF32(
    x: WasmF32Tensor,
    gs: number = JSQ8Tensor.DEFAULT_GROUP_SIZE,
  ): WasmQ8Tensor {
    if (x.length % gs !== 0) {
      throw new Error("Input length must be a multiple of group size");
    }
    const o = WasmQ8Tensor.allocate(x.length, gs);
    quantize(o, x, x.length, gs);
    return o;
  }

  static allocateFromJSF32(
    x: JSF32Tensor,
    gs: number = JSQ8Tensor.DEFAULT_GROUP_SIZE,
  ): WasmQ8Tensor {
    if (x.length % gs !== 0) {
      throw new Error("Input length must be a multiple of group size");
    }
    const o = WasmQ8Tensor.allocate(x.length, gs);
    quantize_js(o, x, x.length, gs);
    return o;
  }

  public readonly gs: number;
  constructor(
    public readonly q: WasmI8Tensor,
    public readonly s: WasmF32Tensor,
  ) {
    super();
    this.gs = this.q.length / this.s.length;
  }

  get [WASM_ADDRESS](): number {
    throw new Error("Not implemented");
  }

  get length() {
    return this.q.length;
  }

  get array(): Int8Array {
    throw new Error("Not implemented");
  }
  private get buffer(): Int8Array {
    throw new Error("Not implemented");
  }

  dequantize(): Float32Array {
    const o = JSF32Tensor.allocate(this.q.length);
    dequantize(o, this, this.length, this.gs);
    return o.array;
  }

  subarray(start: number, end: number): WasmQ8Tensor {
    if (start % this.gs !== 0 || end % this.gs !== 0) {
      throw new Error(
        "Subarray start and end must be a multiple of group size",
      );
    }
    return new WasmQ8Tensor(
      this.q.subarray(start, end),
      this.s.subarray(start / this.gs, end / this.gs),
    );
  }
}
