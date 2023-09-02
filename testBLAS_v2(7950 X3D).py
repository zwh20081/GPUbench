import torch as pt
import torch_musa
import numpy as np
import time as ti
import timeit
import scipy.io as sio

def main():
    
    Nc = 400
    Nmin = 10
    Nmax = 14


    #------------- VV Dot -------------#
    size1 = 10 ** np.arange(3, 10, dtype = int)
    data1 = []

    print('向量点乘性能测试！')

    for L in size1:
        v1 = pt.rand((1, L), dtype = pt.float, device = 'cpu')
        v2 = pt.rand((L, 1), dtype = pt.float, device = 'cpu')

        def getCPUVVDot():
            return v1.dot(v2)

        cpu_time = timeit.timeit(getCPUVVDot, number = Nc)
        print('L='+str(L)+' ,初次运行：')
        print('', 'CPU计算时间:', cpu_time)

        cpu_time = timeit.timeit(getCPUVVDot, number = Nc)
        print('L='+str(L)+' ,再次运行：')
        print('', 'CPU计算时间:', cpu_time)

        data1.append(cpu_time)

    data1 = np.array(data1, dtype = float) / Nc
    #----------------------------------#

    #------------- MV Dot -------------#
    print('矩阵-向量点乘性能测试！')
    size2 =  2 ** np.arange(Nmin, Nmax)
    data2 = []

    for L in size2:
        X1 = pt.rand((L, L), dtype = pt.float, device = 'cpu')
        v1 = pt.rand((L, 1), dtype = pt.float, device = 'cpu')

        def getCPUMVDot():
            return X1.matmul(v1)

        cpu_time = timeit.timeit(getCPUMVDot, number = Nc)
        print('L='+str(L)+' ,初次运行：')
        print('', 'CPU计算时间:', cpu_time)

        cpu_time = timeit.timeit(getCPUMVDot, number = Nc)
        print('L='+str(L)+' ,再次运行：')
        print('', 'CPU计算时间:', cpu_time)

        data2.append(cpu_time)

    data2 = np.array(data2, dtype = float) / Nc
    #----------------------------------#

    #------------- MM MatMul -------------#
    print('矩阵乘法性能测试！')
    size3 =  2 ** np.arange(Nmin, Nmax)
    data3 = []

    for L in size3:
        X1 = pt.rand((L, L), dtype = pt.float, device = 'cpu')
        Y1 = pt.rand((L, L), dtype = pt.float, device = 'cpu')

        def getCPUMMmm():
            return X1.matmul(Y1)

        cpu_time = timeit.timeit(getCPUMMmm, number = Nc)
        print('L='+str(L)+' ,初次运行：')
        print('', 'CPU计算时间:', cpu_time)

        cpu_time = timeit.timeit(getCPUMMmm, number = Nc)
        print('L='+str(L)+' ,再次运行：')
        print('', 'CPU计算时间:', cpu_time)

        data3.append(cpu_time)

    data3 = np.array(data3, dtype = float) / Nc
    #----------------------------------#

    #------------- MM Dot -------------#
    print('矩阵点乘性能测试！')
    size4 =  2 ** np.arange(Nmin, Nmax)
    data4 = []

    for L in size4:
        X1 = pt.rand((L, L), dtype = pt.float, device = 'cpu')
        Y1 = pt.rand((L, L), dtype = pt.float, device = 'cpu')

        def getCPUMMDot():
            return X1.mul(Y1)

        cpu_time = timeit.timeit(getCPUMMDot, number = Nc)
        print('L='+str(L)+' ,初次运行：')
        print('', 'CPU计算时间:', cpu_time)

        cpu_time = timeit.timeit(getCPUMMDot, number = Nc)
        print('L='+str(L)+' ,再次运行：')
        print('', 'CPU计算时间:', cpu_time)

        data4.append(cpu_time)

    data4 = np.array(data4, dtype = float) / Nc
    #----------------------------------#

    #------------- Save Data -------------#
    key = {'size1':size1, 'size2':size2, 'size3':size3, 'size4':size4, \
           'data1':data1, 'data2':data2, 'data3':data3, 'data4':data4,}

    try:
        sio.savemat('data(7950 X3D).mat', key)
        print('数据保存成功！')
    except Exception:
        print('数据保存失败！')

    #------------- Visulization -------------#


if __name__ == '__main__':
    t1 = ti.time()
    main()
    t2 = ti.time()
    print('总用时: ', t2-t1, '秒')
