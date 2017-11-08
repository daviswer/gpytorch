int fft1_r2c_cuda(THCudaTensor *input, THCudaTensor *output);
int fft2_r2c_cuda(THCudaTensor *input, THCudaTensor *output);
int fft3_r2c_cuda(THCudaTensor *input, THCudaTensor *output);
int fft2_c2c_cuda(THCudaTensor *input, THCudaTensor *output);

int fft1_c2r_cuda(THCudaTensor *input, THCudaTensor *output);
int fft2_c2r_cuda(THCudaTensor *input, THCudaTensor *output);
int fft3_c2r_cuda(THCudaTensor *input, THCudaTensor *output);
int ifft2_c2c_cuda(THCudaTensor *input, THCudaTensor *output);

