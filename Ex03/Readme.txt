第三次作业，第1题在“QR_Factorization.py”中。程序import了numpy库、time库（用于计时）和scipy库（用于和标准的QR分解函数比对）。
input输入矩阵的数量和阶数。
输出三种方法的时间，并会验证算法的正确性。

第三次作业，第2题在“Diatomic_Chain_Eigenvalue_Problem.py”。程序import了numpy库。
注：由于初始的矢量是随机的，结果在误差范围内，有一定的随机性。

第三次作业，第3题在“Correlation_Function.py”中。程序import了numpy库、matplotlib库（用于作图）、scipy库（用于求chi-squared分布的p值）。
需要将“correlation-function.dat”与程序放在同一目录下，以导入数据。
依次输出每一小问的结果。

第四次作业，第1题在“Lotka-Volterra.py”中。程序import了numpy库、matplotlib库（用于作图）。
结果除了输出Ex04报告中的Figure 1，还会依次输出每个初值条件下的x-t图、y-t图、x-y图。

第四次作业，第2题在“PDE_by_Iterating.py”中。程序import了numpy库、matplotlib库（如果需要作图）。
运行程序后，先在“N=”时输入N的规模（10、20、50）,再选择是否需要作图（y or n）。
结果会输出三种方法各自的迭代次数和剩余误差r(i)。
注（1）：N=50时费时较长，约40秒左右。
注（2）：由于用幂次法近似求谱半径，而幂次法中初始向量是随机的，每次求出的谱半径略有不同，导致SOR方法的迭代次数略有不同。
