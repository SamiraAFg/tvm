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
* \return error code
*
*/

int vanilla_extended_dense(float* ifmap, float* weights, float* result, int ilen, int olen, int hidden)
{
  /*printf("========================================");
  printf("Dense Layer\n");
  printf("Input size=%dx%d\t Weight size=%dx%d\t Output size=%dx%d\n", hidden, ilen, olen, ilen, hidden, olen); */
  for (int32_t i = 0; i < hidden; ++i) {
    for (int32_t j = 0; j < olen; ++j) {
        for (int32_t k = 0; k < ilen; ++k) {
            int32_t cse_var_1 = ((i * olen) + j);
            if (k == 0) {
                result[cse_var_1] =  0.000000e+00f;
            }
            result[cse_var_1] = (result[cse_var_1] + (ifmap[((i * ilen) + k)] * weights[((j * ilen) + k)]));
        }
    //    printf("result[%d][%d]= %4.2f \n", i, j, result[(i*olen)+j]);
    }
  }
  return 0;
}