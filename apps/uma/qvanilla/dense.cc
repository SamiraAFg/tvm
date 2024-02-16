#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
#endif

 /*!
     * \brief dense function for mock-accelerator examples. Limited to fully-connected layer no bias or activation function. 
     * \param ifmap Pointer to input 2D tensor of size hidden*ilen*sizeof(float). 
     * \param weights Pointer to weight data of size ilen*olen*sizeof(float). 
     * \param result Pointer to output 2D tensor of size hidden*olen*sizeof(float). 
     * \param ilen 
     * \param olen 
     * \param hidden  
     * \param i_zp Input zero point
     * \param k_zp  Filters zero point
     * \return error code
     */
int q_vanilla_accelerator_dense(int8_t* ifmap, int8_t* weights, int32_t* bias_data, int32_t* result, 
                                       int32_t ilen, int32_t olen, int32_t hidden, 
                                       int32_t i_zp, int32_t k_zp)
{
//   printf("========================================");
//   printf("Dense Layer\n");
//   printf("Input size=%dx%d\t Weight size=%dx%d\t Output size=%dx%d\n", hidden, ilen, olen, ilen, hidden, olen); 
  for (int32_t i = 0; i < hidden; ++i) {
    for (int32_t j = 0; j < olen; ++j) {
        for (int32_t k = 0; k < ilen; ++k) {
            int32_t cse_var_1 = ((i * olen) + j);
            if (k == 0) {
                result[cse_var_1] = (int32_t)0;
            }
            result[cse_var_1] = (result[cse_var_1] + ( (int32_t)(ifmap[((i * ilen) + k)] - i_zp) * (int32_t)( weights[((j * ilen) + k)]- k_zp)));
        }
       result[(i*olen)+j] = (int32_t)(result[(i*olen)+j]  + bias_data[(i*olen)+j]); 
    //    printf("result[%d][%d]= %4.2d \n", i, j, result[(i*olen)+j]);
    }
  }
  return 0;
}