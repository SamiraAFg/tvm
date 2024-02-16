#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
#endif

 /*!
* \brief relu function for mock-accelerator examples. 
* \param input Pointer to input 
* \param result Pointer to output 
* \param ilen 
* \return error code
*/
int vanilla_extended_relu(float* input, float* result, int ilen)
{
    // printf("========================================");
    // printf("Relu Layer\n");
    // printf("Input size=%d",  ilen);
    for (int32_t i = 0; i < ilen; ++i) {
        result[i] =  input[i] > 0 ? input[i] : 0;
        // printf("result[%i]= %4.2f \n", i, result[i]);
    }
    return 0;
}