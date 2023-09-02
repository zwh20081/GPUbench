import torch as pt
import torch_musa
import numpy as np
import time as ti
import timeit
import scipy.io as sio

# import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 600
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['text.usetex'] = 'True'

def main():

    Nc = 2
    Nmin = 10
    Nmax = 14
    cpu_time = 0

    #------------- VV Dot -------------#
    print('向量点乘性能测试！')
    size1 = 10 ** np.arange(3, 9, dtype = int)
    data1 = []

    for L in size1:
        v1 = pt.rand((1, L), dtype = pt.float, device = 'musa')
        v2 = pt.rand((L, 1), dtype = pt.float, device = 'musa')

        def getGPUVVDot():
          return pt.matmul(v1, v2)

        gpu_time = timeit.timeit(getGPUVVDot, number = Nc)
        data1.append([gpu_time,cpu_time])

    data1 = np.array(data1, dtype = float)
    #----------------------------------#

    #------------- MV Dot -------------#
    print('矩阵-向量点乘性能测试！')
    size2 =  2 ** np.arange(Nmin, Nmax)
    data2 = []

    for L in size2:
        X1 = pt.rand((L, L), dtype = pt.float, device = 'musa')
        v1 = pt.rand((L, 1), dtype = pt.float, device = 'musa')

        def getGPUMVDot():
          return pt.matmul(X1, v1)

        gpu_time = timeit.timeit(getGPUMVDot, number = Nc)
        data2.append([gpu_time,cpu_time])

    data2 = np.array(data2, dtype = float)
    #----------------------------------#

    #------------- MM MatMul -------------#
    print('矩阵乘法性能测试！')
    size3 =  2 ** np.arange(Nmin, Nmax)
    data3 = []

    for L in size3:
        X1 = pt.rand((L, L), dtype = pt.float, device = 'musa')
        Y1 = pt.rand((L, L), dtype = pt.float, device = 'musa')

        def getGPUMMmm():
          return pt.matmul(X1, Y1)

        gpu_time = timeit.timeit(getGPUMMmm, number = Nc)
        data3.append([gpu_time,cpu_time])

    data3 = np.array(data3, dtype = float)
    #----------------------------------#

    #------------- MM Dot -------------#
    print('矩阵点乘性能测试！')
    size4 =  2 ** np.arange(Nmin, Nmax)
    data4 = []

    for L in size4:
        X1 = pt.rand((L, L), dtype = pt.float, device = 'musa')
        Y1 = pt.rand((L, L), dtype = pt.float, device = 'musa')

        def getGPUMMDot():
          return pt.mul(X1, Y1)

        gpu_time = timeit.timeit(getGPUMMDot, number = Nc)
        data4.append([gpu_time,cpu_time])

    data4 = np.array(data4, dtype = float)
    #----------------------------------#

    key = {'size1':size1, 'size2':size2, 'size3':size3, 'size4':size4, \
           'data1':data1, 'data2':data2, 'data3':data3, 'data4':data4,}

    sio.savemat('data.mat', key)



if __name__ == '__main__':
    t1 = ti.time()
    main()
    t2 = ti.time()
    print('总用时: ', t2-t1, '秒')
