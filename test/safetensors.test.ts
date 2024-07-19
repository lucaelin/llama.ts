import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { bf16ToF32, readHFRepo } from "../src/safetensors.ts";

Deno.test("bf16ToF32 test", () => {
  const a = new Uint8Array([0, 0, 0, 0]);
  assertEquals(bf16ToF32(a), new Float32Array([0, 0]));

  const b = new Uint8Array([0b10000000, 0b00111111, 0b00000000, 0b11000000]);
  assertEquals(bf16ToF32(b), new Float32Array([1, -2]));

  const c = new Uint8Array([0b00000000, 0b00000000, 0b00000000, 0b10000000]);
  assertEquals(bf16ToF32(c), new Float32Array([0, -0]));

  const d = new Uint8Array([0b10000000, 0b01111111, 0b10000000, 0b11111111]);
  assertEquals(bf16ToF32(d), new Float32Array([Infinity, -Infinity]));

  const e = new Uint8Array([0b01001001, 0b01000000, 0b10101011, 0b00111110]);
  //assertEquals(bf16ToF32(d), new Float32Array([Math.PI, 1 / 3]));
  assertEquals(
    bf16ToF32(e),
    new Float32Array([
      3.140625, // Math.PI
      0.333984375, // 1 / 3
    ]),
  );

  const f = new Uint8Array([0xcd, 0x3d, 0x24, 0x3c, 0x83, 0x3a]);
  //assertEquals(bf16ToF32(e), new Float32Array([0.1, 0.01, 0.001]));
  assertEquals(
    bf16ToF32(f),
    new Float32Array([
      0.10009765625, // 0.1
      0.010009765625, // 0.01
      0.00099945068359375, // 0.001
    ]),
  );
});

Deno.test("readHFRepo test", async () => {
  // prepare files
  const testConfig = {};
  const testHeader = {
    "model.layers.0.test.weight": {
      "dtype": "F32",
      "shape": [2, 2],
      "data_offsets": [0, 16],
    },
  };
  const header = new TextEncoder().encode(JSON.stringify(testHeader));
  const headerSize = new Uint8Array(8);
  const headerView = new DataView(headerSize.buffer);
  headerView.setBigUint64(0, BigInt(header.byteLength), true);

  const filecontents: Uint8Array[] = [
    headerSize,
    header,
    new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer),
  ];

  // write to temp files
  const configPath = await Deno.makeTempFile();
  const modelPath = await Deno.makeTempFile();
  const configFileHandle = await Deno.open(configPath, {
    create: true,
    write: true,
  });
  const modelFileHandle = await Deno.open(modelPath, {
    create: true,
    write: true,
  });

  await configFileHandle.write(
    new TextEncoder().encode(JSON.stringify(testConfig)),
  );
  for (const filecontent of filecontents) {
    await modelFileHandle.write(filecontent);
  }
  configFileHandle.close();
  modelFileHandle.close();

  // read from temp files
  const { config, weights } = readHFRepo(configPath, modelPath, "F32");

  assertEquals(config, testConfig);
  assertEquals(weights, {
    model: {
      layers: [{
        test: {
          weight: {
            dtype: "F32" as const,
            shape: [2, 2],
            weights: new Float32Array([1, 2, 3, 4]),
            data_offsets: [0, 16],
          },
        },
      }],
    } as any,
  });
});
