# 使用scipy 求解非线性方程的近似解



## 参考

https://www.jb51.net/article/175045.htm

https://zhuanlan.zhihu.com/p/101645294

https://m.yisu.com/zixun/172658.html

## 背景

最近碰巧需要解方程 $\sin(ax)=bx$, 其中a, b 为常数，需要求x。 乍的一看这个方程非常简单，然而想遍大学的知识也没想出该怎么解这个方程。然后从知乎各位大佬口中得知该方程无解析解，只能通过迭代的方式求出近似解。

## 代码

```python
import numpy as np
from scipy.optimize import root, fsolve


def f(x,a=108,b=99):
    return np.sin(a*x)-b*x

sol1_root = root(f, [2])
sol1_fsolve = fsolve(f, [2])
print(sol1_root)
print(sol1_fsolve)
```

