#include "emscripten.h"
#include "stdint.h"
#include <math.h>

EMSCRIPTEN_KEEPALIVE
void matmul(float* restrict o, const float* restrict x, const float* restrict w, const int n, const int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        #pragma clang loop vectorize(enable)
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        o[i] = val;
    }
}

EMSCRIPTEN_KEEPALIVE
void qmatmul(float* restrict xout, const int8_t* restrict x, const float* restrict xs, const int8_t* restrict w, const float* restrict ws, const int n, const int d, const int GS) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            #pragma clang loop vectorize(enable)
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x[j + k]) * ((int32_t) w[in + j + k]);
            }
            val += ((float) ival) * ws[(in + j) / GS] * xs[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}


EMSCRIPTEN_KEEPALIVE
void quantize(int8_t* restrict o, float* restrict os, const float* restrict x, const int n, const int GS) {
    const int num_groups = n / GS;
    const float Q_MAX = 127.0f;
    
    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        #pragma clang loop vectorize(enable)
        for (int i = 0; i < GS; i++) {
            const float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        const float scale = wmax / Q_MAX;
        os[group] = scale;

        // calculate and write the quantized values
        #pragma clang loop vectorize(enable)
        for (int i = 0; i < GS; i++) {
            const float original_value = x[group * GS + i];
            const float quant_value = original_value / scale; // scale
            const int8_t quantized = (int8_t) round(quant_value); // round and clamp
            o[group * GS + i] = quantized;
        }
    }
}
