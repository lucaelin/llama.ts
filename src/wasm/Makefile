all:
	emcc kernels.c -O3 -ffast-math -fno-protect-parens -msimd128 -Rpass-missed=loop-vectorize --no-entry -Wl,--import-memory -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4294967296 -gsource-map -o kernels.wasm

clean:
	rm -f kernels.wasm kernels.wasm.map