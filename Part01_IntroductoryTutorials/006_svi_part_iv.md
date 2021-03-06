---
jupytext:
  formats: ipynb,md:myst
  text_representation: {extension: .md, format_name: myst, format_version: 0.13, jupytext_version: 1.13.1}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# 变分推断（ $\mathrm{IV}$ ） 技巧和窍门

上面三个变分推断的教程 ([Part I](http://pyro.ai/examples/svi_part_i.html), [Part II](http://pyro.ai/examples/svi_part_ii.html), & [Part III](http://pyro.ai/examples/svi_part_iii.html)) 完成了使用 Pyro 进行变分推断的各步骤。在此过程中，我们定义了`模型`和`引导`（即变分分布）、设置了变分目标（特别是 [ELBOs](https://docs.pyro.ai/en/dev/inference_algos.html?highlight=elbo#module-pyro.infer.elbo)) 、构建了优化器 [pyro.optim](http://docs.pyro.ai/en/dev/optimization.html)。

**所有这些机制的基础是将贝叶斯推断作为随机优化问题**。

这一切都非常有用，但为了达到最终目标（即学习模型参数、推断近似后验、后验预测分布执行预测等），我们需要能够成功地利用 Pyro 来解决这个优化问题。特别是一些细节因素（例如隐空间的维度、是否有离散型隐变量等）往往会给问题带来一定难度。

在本次教程中，介绍了一些在 Pyro 中进行变分推断时非常有用的提示和技巧。例如：如果 ELBO 不收敛怎么办？遇到 `NaN` 时怎么办？ 下面的内容可能对你有帮助！

> **Pyro 论坛**
> 如果你在阅读了本教程后，仍然存在优化问题，可以毫不犹豫地到 [Pyro 论坛](https://forum.pyro.ai/) 提问!

+++

## 技巧 1 ： 从小的学习率开始

虽然较大的学习率可能适用于某些问题，但通常最好从较小的学习率开始，例如 $10^{-3}$ 或 $10^{-4}$：

```python
optimizer = pyro.optim.Adam({"lr": 0.001})
```

这是因为 ELBO 梯度是**随机的**，并且具有潜在的高方差，因此大的学习率会迅速导致`模型`和`引导`的参数空间区域在数值上不稳定或在其他方面受影响。

一旦你使用较小的学习率实现了稳定的 ELBO 优化，就可以尝试更大的学习率。这通常是一个好主意，因为过小的学习率也会导致优化效果不佳。特别是小的学习率存在陷入 ELBO 局部最优的可能性。

+++

## 技巧 2 ： 默认使用 Adam 或 ClippedAdam 优化器

在做随机变分推断时，建议默认使用 [Adam](http://docs.pyro.ai/en/stable/optimization.html?highlight=clippedadam#pyro.optim.pytorch_optimizers.Adam) 或 [ClippedAdam](http://docs.pyro.ai/en/stable/optimization.html?highlight=clippedadam#pyro.optim.optim.ClippedAdam) 。 请注意，`ClippedAdam` 只是`Adam` 的一个方便的扩展，它提供了对学习率衰减和梯度裁剪的内置支持。

这些优化算法通常在变分推断的上下文中表现良好，其基本原因是，当优化问题非常随机时，通过参数的动量来提供平滑的梯度通常必不可少。注意：在随机变分推断中，随机性可以来自隐变量的采样、数据的二次采样或两者都有。

除了在某些情况下调整学习率之外，可能还需要调整用于控制 `Adam` 动量的超参数 `betas` 。特别是对于非常随机的模型，使用更高的 $\beta_1$ 值可能更有意义（注：`betas` 是一对超参数），例如：

用

```python
betas = (0.95, 0.999)
```
代替

```python
betas = (0.90, 0.999)
```

+++

## 技巧 3 ：考虑使用逐步衰减的动态学习率

虽然在优化开始时，当你离最优值还很远并且想要采取较大的梯度步长时，适度大的学习率比较有用，但稍后使用较小的学习率通常也很有用，这样你就不会在在不收敛的情况下过度优化。

实现这种动态调整的一种方法是使用 [Pyro 提供的学习率调度器](http://docs.pyro.ai/en/stable/optimization.html?highlight=scheduler#pyro.optim.lr_scheduler.PyroLRScheduler) 。例子可以参见[这里的代码片段](https://github.com/pyro-ppl/pyro/blob/a106882e8ffbfe6ac96f19aef9a218026482ed51/examples/scanvi/scanvi.py#L265) 。

另外一种比较便利的方法，是使用前面提到的 [ClippedAdam 优化器](http://docs.pyro.ai/en/stable/optimization.html?highlight=clippedadam#pyro.optim.optim.ClippedAdam)  ， 它可以通过 `lrd` 参数来控制学习率的衰减:

```python
num_steps = 1000
initial_lr = 0.001
gamma = 0.1  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / num_steps)
optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
```

+++

## 技巧 4 ： 确保`模型`和`引导`具有同样的概率分布约束

假设你的 `model` 中有一个具有某种约束的分布，例如一个 `LogNormal` 分布，它仅支持正实轴：

```python
def model():
    pyro.sample("x", dist.LogNormal(0.0, 1.0))
``` 
那么你必须保证在 `guide` 中的配套 `sample` 中具有相同的约束支持：

```python
def good_guide():
    loc = pyro.param("loc", torch.tensor(0.0))
    pyro.sample("x", dist.LogNormal(loc, 1.0))
``` 
如果你没有这样做，而是使用了如下不可受理的 `guide`：

```python
def bad_guide():
    loc = pyro.param("loc", torch.tensor(0.0))
    # Normal may sample x < 0
    pyro.sample("x", dist.Normal(loc, 1.0))  
```

你会快速、大概率地进入 `NaN` 状态。这是因为 `LogNormal` 分布在小于 0 的样本 `x` 处的 `log_prob` 没有定义，而这个 `bad_guide` 依然想生成这个样本。

+++

## 技巧 5 ：用 `constraint` 来约束需要约束的参数

同样，你需要确保用于创建某个分布实例的参数有效，否则你很快就会遇到 `NaN` 。例如，正态分布的 `scale` 参数必须为正。因此，以下的`bad_guide` 是有问题的：

```python
def bad_guide():
    scale = pyro.sample("scale", torch.tensor(1.0))
    pyro.sample("x", dist.Normal(0.0, scale))
``` 
而一个好的 `guide` 应当使用 `constraint` 以确保其为正值:
```python
from pyro.distributions import constraints

def good_guide():
    scale = pyro.sample("scale", torch.tensor(0.05),               
                        constraint=constraints.positive)
    pyro.sample("x", dist.Normal(0.0, scale))
```

+++

## 技巧 6 ：如果在构建`引导`时遇到问题，建议使用 AutoGuide

为了使`模型/引导`对能够产生稳定的优化，需要满足许多条件，我们在之前教程中介绍了其中一些条件。有时非常难以诊断数值不稳定或不收敛的原因。因为问题可能出现在许多不同的地方：在模型中、在引导中、或者在优化算法或超参数的选择中。

有时你认为问题出在`引导`中，而实际上有可能出在你的`模型`中。相反，有时问题出在`引导`中，但你可能认为问题出在`模型`或其他地方。

出于此原因，在尝试确定底层问题时减少活动部件的数量可能会有所帮助。一种便利的方法是将自定义的`引导`替换为 [pyro.infer.AutoGuide](http://docs.pyro.ai/en/stable/infer.autoguide.html#module-pyro.infer.autoguide) 。

例如，如果模型中的所有隐变量都是连续的，你可以尝试使用 [pyro.infer.AutoNormal](http://docs.pyro.ai/en/stable/infer.autoguide.html#autonormal) 做为`引导`。

或者，你可以使用 MAP 推断而不是变分推断。有关更多详细信息，请参阅 [MLE/MAP 教程](http://pyro.ai/examples/mle_map.html) 。一旦 MAP 推断能够正常工作，你就有充分理由相信模型设置没问题（至少就基本数值稳定性而言）。

如果你对获得近似后验分布感兴趣，现在可以使用成熟的随机变分推断进行跟进。实际上，一个自然的操作顺序可能会使用以下越来越灵活的自动引导：

[AutoDelta](http://docs.pyro.ai/en/stable/infer.autoguide.html#autodelta)   →  [AutoNormal](http://docs.pyro.ai/en/stable/infer.autoguide.html#autonormal)  →  [AutoLowRankMultivariateNormal](http://docs.pyro.ai/en/stable/infer.autoguide.html#autolowrankmultivariatenormal)

如果你发现需要更灵活的引导，或者想要更好地控制引导的定义方式，那么可以继续构建自定义的引导。一种方法是利用 [easy guides](http://pyro.ai/examples/easyguide.html) ，它在完全自定义和自动化之间取得一定的平衡。

需注意，自动引导提供了多种初始化策略，在某些情况下需要对这些策略进行试验才能获得良好的优化性能。

控制初始化行为的一种方法是使用 `init_loc_fn` 。关于 `init_loc_fn` 和 `easy guide` 的示例用法，可以参阅 [这里](https://github.com/pyro-ppl/pyro/blob/a106882e8ffbfe6ac96f19aef9a218026482ed51/examples/sparse_gamma_def.py#L202) 。

+++

## 技巧 7 ：参数的初始化非常重要，好的初始化会减小方差

优化问题中的初始化方法会导致“好方案和灾难性失败”之间所有可能差异。很难为初始化提出一套全面而良好的实践，因为好的初始化方案通常非常依赖于问题本身。

在随机变分推断的上下文中，初始化引导以使其具有**低方差**通常是一个好主意。这是因为用于优化 ELBO 的梯度是随机梯度。如果在 ELBO 优化开始时的梯度由于初始化不良而表现出高方差，那么优化过程极有可能会被引入数值不稳定或其他不好的参数空间区域中。

防范此危险的一种方法是密切注意`引导`中控制方差的参数。例如，通常希望下面合理初始化的引导：

```python
from pyro.distributions import constraints

def good_guide():
    scale = pyro.sample("scale", torch.tensor(0.05),               
                        constraint=constraints.positive)
    pyro.sample("x", dist.Normal(0.0, scale))
``` 

同时下面的高方差引导极大可能会导致问题：

```python
def bad_guide():
    scale = pyro.sample("scale", torch.tensor(12345.6),               
                        constraint=constraints.positive)
    pyro.sample("x", dist.Normal(0.0, scale))
``` 

> **注意**
>
> 各种自动引导的初始方差都可以用 `init_scale` 参数来控制, 可参阅 [关于 AutoNormal 的示例](http://docs.pyro.ai/en/stable/infer.autoguide.html?highlight=init_scale#autonormal) 。

+++

## 技巧 8 ：通过粒子数量、批次大小等参数来做权衡

如果你的 ELBO 表现出很大方差，优化可能会很困难。你可以尝试增加用于计算每个随机 ELBO 估计的粒子数：

```python
elbo = pyro.infer.Trace_ELBO(num_particles=10, 
                             vectorize_particles=True)
```

> 注意，要使用 `vectorized_pa​​rticles=True` 参数设置，并且确保`模型`和`引导`被正确的向量化；请参阅 [张量形状教程](http://pyro.ai/examples/tensor_shapes.html) 。

另外一种策略是用时间效率换取低方差梯度。如果你正在进行数据二次采样，小批量的大小提供了类似的权衡：大的批量以更多计算为代价减少了方差。

尽管最好的方法取决于问题，但通常值得用更少的粒子采取更多的梯度步骤，而不是用更多粒子采用更少的梯度步骤。

一个重要的提醒是当你在 GPU 上运行时，对于某些模型增加粒子数或批量大小的成本可能是次线性的，在这种情况下增加粒子数可能更有吸引力。

+++

## 9. 如果可用的话，可以使用 `TraceMeanField_ELBO` 

Pyro 中一个基础的 `ELBO` 实现是 [Trace_ELBO](http://docs.pyro.ai/en/stable/inference_algos.html?highlight=tracemeanfield#pyro.infer.trace_elbo.Trace_ELBO) ， 使用随机样本估计 KL 散度项。

当 KL 散度具有解析式时，你或许可以使用解析的 KL 散度来减小 ELBO 的方差。这个功能由 [TraceMeanField_ELBO](http://docs.pyro.ai/en/stable/inference_algos.html?highlight=tracemeanfield#pyro.infer.trace_elbo.Trace_ELBO) 提供。

+++

### 10. 考虑标准化你的 ELBO

默认情况下，Pyro 计算一个非标准化的 ELBO，即它计算的值是在完整数据集上计算的对数证据下界。

对于大型数据集，这可能是一个非常大的数。

由于计算机使用有限精度（例如 32 位浮点数）进行算术运算，因此大数可能会影响数值稳定性，因为它们会导致精度损失、下溢/溢出等问题。

因此，在许多情况下使 ELBO 标准化，使其大致归一，会有所帮助。

这同时也有助于粗略地了解 ELBO 值的优劣。

例如，如果我们有维度为 $D$ 的 $N$ 个数据点（维度为 $D$ 的 $N$ 个实值向量），那么通常期望一个合理优化的 ELBO 是 $N\times D$ 阶的。

因此，如果将 ELBO 重新被一个 $N \times D$ 的因子归一化，那我们期望其结果最好是一个一阶 ELBO 。

> 此处没看懂。

虽然这只是一个粗略的经验法则，但如果使用这种标准化能够得到 $-123.4$ 或 $1234.5$ 等 ELBO 值，那么会提醒我们可能存在问题：也许模型被严重错误指定了；也许初始化非常糟糕等等。

有关如何通过归一化常量扩展 ELBO 的详细信息，请参阅 [本教程](http://pyro.ai/examples/custom_objectives.html#Example:-Scaling-the-Loss ）。

+++

## 11. 注意尺度问题

数字的尺度很重要。

它们之所以重要，至少有两个重要原因：

i) 尺度可以成就或破坏特定的初始化方案； 

ii) 如上一节所述，尺度会对数值精度和稳定性产生影响。

为了具体说明，假设你正在进行线性回归，即正在学习 $Y = WX$ 形式的线性映射。通常数据带有特定的单位。

例如，协变量 $X$ 的某些组成部分可能以美元为单位（例如房价），而其他部分可能以密度为单位（例如每平方英里的居民数）。也许第一个协变量具有典型值，如 $10^5$，而第二个协变量具有典型值，如 $10^2$。

当你遇到跨越多个数量级的数字时，你应该始终注意。

在许多情况下，将事物规范化以使它们成为有序统一体是有意义的。例如，你可以以 100,000 美元为单位来衡量房价。

这些类型的数据转换可以为下游建模和推断带来许多好处。例如，如果你已适当地对所有协变量进行了标准化，则在权重上设置简单的各向同性先验可能是合理的。

```python
pyro.sample("W", dist.Normal(torch.zeros(2), torch.ones(2)))
```
而不是必须为不同协变量指定不同的先验协方差。

```python
prior_scale = torch.tensor([1.0e-5, 1.0e-2])
pyro.sample("W", dist.Normal(torch.zeros(2), prior_scale))
```

还有其他好处。现在可以更轻松地为你的引导函数初始化适当的参数。 [pyro.infer.AutoGuide](http://docs.pyro.ai/en/stable/infer.autoguide.html#module-pyro.infer.autoguide) 使用的默认初始化现在也更有可能解决你的问题。

+++

## 12. 保持启用验证

默认情况下，Pyro 会启用验证逻辑，这对调试`模型`和`引导`很有帮助。例如，当分布参数无效时，验证逻辑会通知你。除非你有很好的理由，否则请尽量保持启用验证逻辑。一旦你对一个模型和推断过程感到满意，你可以使用 [pyro.enable_validation](http://docs.pyro.ai/en/stable/primitives.html?highlight=enable_validation#pyro.primitives.enable_validation) 禁用验证逻辑。

同样在`ELBOs`的上下文中，当隐变量是枚举型离散变量时，做如下设置也许是正确的选择：

```python
strict_enumeration_warning=True
```
when you are enumerating discrete latent variables.

+++

## 13. 张量形状错误

如果你遇到张量形状错误，请确保你已仔细阅读[相应教程](http://pyro.ai/examples/tensor_shapes.html)。

+++

## 14. 如果可能，枚举离散型隐变量

如果你的模型包含离散型隐变量，则精确地枚举它们可能是有意义的，因为这可以显着减少 ELBO 的方差。更多讨论见[对应教程](http://pyro.ai/examples/enumeration.html)。

+++

## 15. 一些复杂的模型可以从 KL 退火中受益

ELBO 的特定形式编码了 “通过对数似然项拟合模型” 和 “通过 KL 散度的先验正则化”之间的权衡。

在某些情况下，KL 散度可以作为一个障碍，使其很难找到好的最优解。在这些情况下，在优化过程中对 KL 散度项的相关强度进行退火会有所帮助。如需进一步讨论，请参阅 [深度马尔可夫模型教程](http://pyro.ai/examples/dmm.html#The-Black-Magic-of-Optimization)。

+++

## 16. 考虑防御性地做梯度裁剪或参数约束

模型或引导中的某些参数可能会控制某些对数值敏感的分布参数。例如，定义 [Gamma](http://docs.pyro.ai/en/stable/distributions.html#gamma) 分布的 `concentration` 和 `rate` 参数可能表现出这种敏感性。

在这些情况下，防御性地裁剪梯度或约束参数可能是有意义的。请参阅 [此代码片段](https://github.com/pyro-ppl/pyro/blob/dev/examples/sparse_gamma_def.py#L135) 以获取渐变裁剪的示例。对于“防御性”参数约束的简单示例，请考虑 `Gamma` 分布的 `concentration` 参数。此参数必须为正：`concentration > 0`。如果想确保 `concentration` 远离零，可以使用带有适当约束的 `param` 语句：

```python
from pyro.distributions import constraints

concentration = pyro.param("concentration", torch.tensor(0.5),
                           constraints.greater_than(0.001))
```

这些技巧可以帮助确保`模型`和`引导`远离参数数值空间中的危险区域。
