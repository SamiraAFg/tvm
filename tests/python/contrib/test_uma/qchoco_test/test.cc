#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>

#include <stdio.h>

#ifdef __cplusplus
extern "C"
#endif



void DeviceCfg(int8_t* a, int8_t* b, int32_t* out, int32_t inp_zp, int32_t wgt_zp, void* streamPointer, size_t streamSize) {

    // std::cout << "DeviceCfg" << std::endl;

  int batch = 1;
  int block_in = 16;
  int block_out = 16;

  for (uint32_t i=0; i < batch; ++i) {
    for (uint32_t j=0; j < block_out; ++j) {
        out[i*block_out + j] = 0;
        // std::cout << "=========" << std::endl;
        for (uint32_t k=0; k < block_in; ++k) {
            
            out[i*block_out + j] += (a[i*block_in + k] - inp_zp) * (b[j*block_in + k] - wgt_zp);
            // std::cout << c[i*block_out + j] << std::endl;
        }
        // std::cout << "c[" << j << "] = " << c[i*block_out + j] << std::endl;
    }
    }
}

int DeviceRun() {

    return 0;
}
