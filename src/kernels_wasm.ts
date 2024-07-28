import { WASM_ADDRESS, WasmF32Tensor, WasmQ8Tensor } from "./types_wasm.ts";

export let memory: WebAssembly.Memory;
export let instance: WebAssembly.Instance;
export function reset() {
  memory = new WebAssembly.Memory({
    initial: 258, // pages of 64KB
    maximum: 4294967296 / (64 * 1024),
  });

  const module = new WebAssembly.Module(
    Deno.readFileSync("./src/wasm/kernels.wasm"),
  );
  instance = new WebAssembly.Instance(module, { env: { memory } });
}

const WASM_PAGE_SIZE = 64 * 1024;
let allocation_pointer = 0;
const allocation_alignment = 8;

export function allocate(bytes: number, memory: WebAssembly.Memory): number {
  const pointer = allocation_pointer;
  allocation_pointer += bytes;
  const alignment_offset = allocation_alignment -
    (allocation_pointer % allocation_alignment);
  allocation_pointer += alignment_offset;
  const grow = allocation_pointer - memory.buffer.byteLength;
  if (grow > 0) {
    memory.grow(Math.ceil(grow / WASM_PAGE_SIZE));
  }
  return pointer;
}

reset();

export function matmul(
  o: WasmF32Tensor,
  x: WasmF32Tensor,
  w: WasmF32Tensor,
  n: number,
  d: number,
): void {
  const { matmul } = instance.exports as { [key: string]: CallableFunction };
  matmul(o[WASM_ADDRESS], x[WASM_ADDRESS], w[WASM_ADDRESS], n, d);
}

export function qmatmul(
  o: WasmF32Tensor,
  x: WasmQ8Tensor,
  w: WasmQ8Tensor,
  n: number,
  d: number,
  gs: number,
): void {
  const { qmatmul } = instance.exports as { [key: string]: CallableFunction };
  qmatmul(
    o[WASM_ADDRESS],
    x.q[WASM_ADDRESS],
    x.s[WASM_ADDRESS],
    w.q[WASM_ADDRESS],
    w.s[WASM_ADDRESS],
    n,
    d,
    gs,
  );
}

export function quantize(
  o: WasmQ8Tensor,
  x: WasmF32Tensor,
  n: number,
  gs: number,
): void {
  const { quantize } = instance.exports as { [key: string]: CallableFunction };
  quantize(
    o.q[WASM_ADDRESS],
    o.s[WASM_ADDRESS],
    x[WASM_ADDRESS],
    n,
    gs,
  );
}
