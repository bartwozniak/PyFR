# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict

class CustomPyFRKernelGenerator:

    def __aggregateSubterms(self, matrix, double=True):
        """ Create a dictionary, which maps each row in matrix to another 
            dictionary. The inner dictionary maps each non-zero value in matrix 
            to the column index at which it occurs. """

        rowData = defaultdict(lambda: defaultdict(list));
        it = np.nditer(matrix, flags=['multi_index', 'refs_ok']);
        while not it.finished:
            if double:
                if float(it[0].item()) != 0.0:
                    rowData[it.multi_index[0]][it[0].item()].append(it.multi_index[1]);
            else:
                if np.float32(it[0].item()) != 0.0:
                    rowData[it.multi_index[0]][it[0].item()].append(it.multi_index[1]);
            it.iternext();

        return rowData;


    def __generateInnerProduct(self, data, double=True):
        """ Return a list of statements in the inner loop for matrix 
            multiplication, fully unrolled """

        # subterms is a mapping from the string subterm sum to a variable name
        subterms, subtermCount = {}, 0;
        # products is a mapping from the row index to a product of subterms 
        # with non-zero values from m
        products = {};
        # for each row in m we will have an output value
        for rowIndex in data.keys():
            rowProducts = [];
            # for every non-zero value in the row of m we need to do a 
            # multiplication with the sum of terms in x
            for nzval, termIndices in data[rowIndex].iteritems():
                rowSubterms = [];
                for termIndex in termIndices:
                    rowSubterms.append('b_local[{index} * bstride]'
                        .format(index = termIndex));
                subterm = ' + '.join(rowSubterms);

                if subterm not in subterms.keys():
                    subterms[subterm] = 'subterm_' + str(subtermCount);
                    subtermCount += 1;

                if double:
                    productSubtermTmpl = '{nzval} * {subterm}'
                else:
                    productSubtermTmpl = '{nzval}f * {subterm}'
                rowProducts.append(productSubtermTmpl
                    .format(nzval = nzval, subterm = subterms[subterm]));

            products[rowIndex] = ' + '.join(rowProducts);
        

        printSubterms, printProducts = [], [];
        if double:
            subtermTmpl = 'const double {var} = {sum};'
        else:
            subtermTmpl = 'const float {var} = {sum};'
        productTmpl = 'c_local[{index} * cstride] = {product};'
        for subterm in subterms.keys():
            printSubterms.append(subtermTmpl
                .format(var = subterms[subterm], sum = subterm));

        for rowIndex, product in products.iteritems():
            printProducts.append(productTmpl
                .format(index = rowIndex, product = product));

        if double:
            local = ['const double *b_local = b + index;', 
                    'double *c_local = c + index;'];
        else:
            local = ['const float *b_local = b + index;', 
                    'float *c_local = c + index;'];
        return local + printSubterms + printProducts;


    def __formatOuterKernel(self, innerProduct, double=True):
        """ Formats a c source file with a matrix multiplication kernel 
            generated for a particular matrix m. """

        if double:
            stub = "__global__ void MatMulKernel(const double* __restrict__ b, double* __restrict__ c, const int bwidth, const int bstride, const int cstride)";
        else:
            stub = "__global__ void MatMulKernel(const float* __restrict__ b, float* __restrict__ c, const int bwidth, const int bstride, const int cstride)";

        loop  = "\tint index = blockDim.x * blockIdx.x + threadIdx.x;"
        loop += "\n\tif (index < bwidth)"

        methodbody =  '{\n' + loop + '\n\t{';
        methodbody += '\n\t\t'.join([''] + innerProduct);
        methodbody += '\n\t}\n' + '}\n';

        return stub + '\n' + methodbody;

    def formatKernel(self, matrix, double=True):
        preprocessedData = self.__aggregateSubterms(matrix, double)
        innerProduct = self.__generateInnerProduct(preprocessedData, double)
        kernelCode = self.__formatOuterKernel(innerProduct, double)

        return kernelCode


    def splay(self, matrix, outWidth):
        from pycuda.tools import DeviceData
        import pycuda.driver as drv

        dev = drv.Context.get_device()
        devdata = DeviceData(dev)
        
        max_block_dim_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X) 
        max_grid_dim_x = dev.get_attribute(drv.device_attribute.MAX_GRID_DIM_X)

        if outWidth > max_block_dim_x * max_grid_dim_x:
            print 'CUDA cannot handle matrices that big'
            raise ValueError

        min_threads = devdata.warp_size
        max_threads = 256
        max_blocks = 4 * devdata.thread_blocks_per_mp \
                * dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)

        if outWidth < min_threads:
            block_count = 1
            threads_per_block = min_threads
        elif outWidth < max_blocks * min_threads:
            block_count = (outWidth + min_threads - 1) // min_threads
            threads_per_block = min_threads
        elif outWidth < max_blocks * max_threads:
            block_count = max_blocks
            grp = (outWidth + min_threads - 1) // min_threads
            threads_per_block = ((grp + max_blocks - 1) // max_blocks) * min_threads
        else:
            threads_per_block = max_threads
            block_count = (outWidth + max_threads - 1) // max_threads

        return (block_count, 1), (threads_per_block, 1, 1)
