第二次作业说明：
共三个文件：Runge_Phenomenon.py，CubicSpline_Plot.py，ZetaFunction.py。
由于方便作图起见，三个文件都
import math
import numpy as np
import matplotlib.pyplot as plt
其中只用numpy图一些方便，如linspace等函数。

可以用IDLE或记事本Edit查看代码。

双击运行，Runge_Phenomenon.py给出原函数以及三种拟合方法给出的图像；
CubicSpline_Plot.py给出各段S(X;t)与S(Y;t)的函数，以及原心型线和样条插值图像。
ZetaFunction.py先给出Zeta00函数在(0,3)上的图像，并在计算每一个点的函数值时输出(n^2)取到多少；再给出第二小题方程的解。