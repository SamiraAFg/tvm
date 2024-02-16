#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
#endif

 /*!
     * \brief clip/relu function for mock-accelerator examples.  
     * \param input Pointer to input 
     * \param result Pointer to output 
     * \param ilen 
     * \return error code
     */
int q_vanilla_accelerator_relu(int8_t* input, int8_t* result, int ilen)
{
    // printf("========================================");
    // printf("Relu Layer\n");
    // printf("Input size=%d",  ilen);
    for (int32_t i = 0; i < ilen; ++i) {
        result[i] =  input[i] < (int8_t)127 ? input[i] : (int8_t)127;
        result[i] =  result[i] > (int8_t)-128 ? result[i] : (int8_t)-128;
      //  printf("result[%i]= %4.2f \n", i, result[i]);
    }
    return 0;
}