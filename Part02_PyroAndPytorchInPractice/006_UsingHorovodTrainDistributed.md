# 案例: 通过 Horovod 做分布式训练

与其他例子不同，本案例必须在[horovodrun](https://github.com/horovod/horovod/blob/master/docs/running.rst) 中运行。

例如：

``` none
$ horovodrun -np 2 examples/svi_horovod.py
```
Pyro 中唯一与 Horovod 相关的组件是 [HorovodOptimizer class](https://docs.pyro.ai/en/latest/optimization.html#pyro.optim.horovod.HorovodOptimizer) 。


[源代码见 Github 上的 svi_horovod.py](https://github.com/pyro-ppl/pyro/blob/dev/examples/svi_horovod.py) 。

