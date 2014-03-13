# -*- coding: utf-8 -*-

import numpy as np
from pyfr.backends.cuda.provider import CUDAKernelProvider
from pyfr.backends.base import ComputeKernel, traits

import bemmGenerator as gen

class CustomPyFRKernels(CUDAKernelProvider):

    GEMM_NAME = 'MatMulKernel'
    GEMM_ARGTYPES = [np.intp, np.intp, np.int32, np.int32, np.int32]
    
    def __init__(self, backend):
        import pycuda.autoinit
        if pycuda.autoinit.context:
            pass


    def __del__(self):
        import pycuda.autoinit
        if pycuda.autoinit.context:
            pass


    @traits(a={'dense'})
    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        if alpha != 1.0 or beta != 0.0:
            print 'Can not handle case of: ', a.tags, alpha, beta, 'Defaulting to cublas'
            raise NotImplementedError
 
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a * b')
    
        # Generate the kernel
        generator = gen.CustomPyFRKernelGenerator()
        matrix = a.get()
        src = generator.formatKernel(matrix)
        #grid, block = generator.splay(matrix, out.ncol)
    
        # Build
        kern = self._build_kernel(self.GEMM_NAME, src, self.GEMM_ARGTYPES)

        from math import ceil
        # Compute a suitable block and grid
        num_blocks = int(ceil(out.ncol / 32.0))
        grid, block = (num_blocks, 1), (32, 1, 1)
        
        class MulKernel(ComputeKernel):
            def run(iself, scomp, scopy):
                kern.prepared_async_call(grid, block, scomp, b, out, b.ncol, b.leaddim, out.leaddim)
    
        return MulKernel()
