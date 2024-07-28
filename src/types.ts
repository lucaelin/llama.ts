import { dequantize, quantize } from "./kernels.ts";

declare global {
  interface ArrayBuffer {
    [Symbol.metadata]: { name: "ArrayBuffer" };
  }
}

export abstract class Tensor {
  static allocate(_n: number): Tensor {
    throw new Error("Not implemented");
  }
  static allocateFrom(_x: number[]) {
    throw new Error("Not implemented");
  }
  abstract get array(): ArrayLike<number>;
  abstract subarray(start: number, end: number): Tensor;
  abstract length: number;
}

export interface F32Tensor {
  get array(): Float32Array;
  subarray(start: number, end: number): F32Tensor;
  length: number;
}

export class JSF32Tensor extends Tensor implements F32Tensor {
  static allocate(n: number) {
    return new JSF32Tensor(new Float32Array(n));
  }
  static allocateFrom(x: number[]) {
    const array = new Float32Array(x.length);
    array.set(x);
    return new JSF32Tensor(array);
  }

  constructor(
    private buffer: Float32Array,
  ) {
    super();
  }

  get length(): number {
    return this.buffer.length;
  }

  get array() {
    return this.buffer;
  }

  subarray(start: number, end: number): JSF32Tensor {
    return new JSF32Tensor(
      this.buffer.subarray(
        start,
        end,
      ),
    );
  }
}

export interface I8Tensor {
  get array(): Int8Array;
  subarray(start: number, end: number): I8Tensor;
  length: number;
}

export class JSI8Tensor extends Tensor implements I8Tensor {
  static allocate(n: number) {
    return new JSI8Tensor(new Int8Array(n));
  }
  static allocateFrom(x: number[]) {
    const buffer = new Int8Array(x.length);
    buffer.set(x);
    return new JSI8Tensor(buffer);
  }

  constructor(
    private buffer: Int8Array,
  ) {
    super();
  }

  get length(): number {
    return this.buffer.length;
  }

  get array() {
    return this.buffer;
  }

  subarray(start: number, end: number): JSI8Tensor {
    return new JSI8Tensor(
      this.buffer.subarray(
        start,
        end,
      ),
    );
  }
}

export interface U8Tensor {
  get array(): Uint8Array;
  subarray(start: number, end: number): U8Tensor;
  length: number;
}

export class JSU8Tensor extends Tensor implements U8Tensor {
  static allocate(n: number) {
    return new JSU8Tensor(new Uint8Array(n));
  }
  static allocateFrom(x: number[]) {
    const buffer = new Uint8Array(x.length);
    buffer.set(x);
    return new JSU8Tensor(buffer);
  }

  constructor(
    private buffer: Uint8Array,
  ) {
    super();
  }

  get length(): number {
    return this.buffer.length;
  }

  get array() {
    return this.buffer;
  }

  subarray(start: number, end: number): JSU8Tensor {
    return new JSU8Tensor(
      this.buffer.subarray(
        start,
        end,
      ),
    );
  }
}

export interface Q8Tensor {
  subarray(start: number, end: number): Q8Tensor;
  length: number;
  q: I8Tensor;
  s: F32Tensor;
  gs: number;
}

export class JSQ8Tensor extends Tensor implements Q8Tensor {
  static readonly DEFAULT_GROUP_SIZE = 32; // Andrej uses 64, according to thebloke 128 is optimal, the current implementation degrades above 32 for small models

  static allocate(
    n: number,
    gs: number = JSQ8Tensor.DEFAULT_GROUP_SIZE,
  ): JSQ8Tensor {
    return new JSQ8Tensor(
      JSI8Tensor.allocate(n),
      JSF32Tensor.allocate(n / gs),
    );
  }
  static allocateFrom(
    x: number[],
    gs: number = JSQ8Tensor.DEFAULT_GROUP_SIZE,
  ): JSQ8Tensor {
    if (x.length % gs !== 0) {
      throw new Error("Input length must be a multiple of group size");
    }
    const o = JSQ8Tensor.allocate(x.length, gs);
    quantize(o, JSF32Tensor.allocateFrom(x), x.length, gs);
    return o;
  }
  static allocateFromF32(
    x: JSF32Tensor,
    gs: number = JSQ8Tensor.DEFAULT_GROUP_SIZE,
  ): JSQ8Tensor {
    if (x.length % gs !== 0) {
      throw new Error("Input length must be a multiple of group size");
    }
    const o = JSQ8Tensor.allocate(x.length, gs);
    quantize(o, x, x.length, gs);
    return o;
  }

  public readonly gs: number;
  constructor(public readonly q: JSI8Tensor, public readonly s: JSF32Tensor) {
    super();
    this.gs = q.array.length / s.array.length;
  }

  get array(): ArrayLike<number> {
    throw new Error("Not implemented");
  }

  public get length(): number {
    return this.q.length;
  }

  public dequantize(): Float32Array {
    const n = this.length;
    const o = JSF32Tensor.allocate(n);
    dequantize(o, this, n, this.gs);
    return o.array;
  }

  public subarray(start: number, end: number): JSQ8Tensor {
    if (start % this.gs !== 0 || end % this.gs !== 0) {
      throw new Error(
        "Subarray start and end must be a multiple of group size",
      );
    }
    return new JSQ8Tensor(
      this.q.subarray(start, end),
      this.s.subarray(start / this.gs, end / this.gs),
    );
  }
}
