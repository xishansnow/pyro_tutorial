---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 半监督变分自编码器（ The Semi-Supervised VAE ）

## 1 概述

我们在教程中介绍的大多数模型都是无监督的：

- [变分自编码器 ( VAE )](001_vae.ipynb)
- [深度马尔科夫模型（ DMM ）](005_dmm.ipynb)
- [Attend-Infer-Repeat](006_air.ipynb)

我们也覆盖了一些简单的监督模型：

- [贝叶斯回归（ Bayesian Regression ）](001_bayesian_regression.ipynb)

半监督设置代表了一种有趣的中间情况，其中一些数据被有标签，而另外一些没有。此类数据具有重要的实际意义，因为通常只有很少的有标签数据和更多的无标签数据，显然我们希望利用有标签数据来改进无标签数据的模型。

半监督设置也非常适合生成模型，其中缺失的数据可以很自然地解释。正如即将看到的，当聚焦在半监督生成模型上时，不乏各种模型变体和可能的推断策略。

尽管我们只能详细探讨其中的一部分，但希望你在结束本教程后能更好地了解概率编程提供的抽象和模块化。

让我们开始构建生成模型。

现在有一个包含  $N$ 个数据点的数据集 $\mathcal{D}$ ，$\mathcal{D} = \{ ({\bf x}_i, {\bf y}_i) \}$，其中 $\{ {\bf x}_i \}$ 总是具有观测值，而标签 $\{ {\bf y}_i \}$ 中，只有一部分有观测值。 因此我们希望能够对这种复杂数据建模。我们计划建立一个隐变量模型，其中局部隐变量 ${\bf z}_i$ 对每对 $({\bf x}_i, {\bf y}_i)$ 都是私有的。即使做了这种假设，仍然存在许多可能的模型，其中我们将重点关注图 1 中描绘的模型（这是参考文献 [1] 中的模型 M2）。

<img src="http://pyro.ai/examples/_static/img/ss_vae_m2.png" style="zoom:33%;" />

> **图 1：** 我们的半监督生成模型（参见参考文献 [1] 中的模型 M2）

为方便起见（同时由于将在下面实验中对 MNIST 进行建模），让我们假设 $\{ {\bf x}_i \}$ 是图像，而 $\{ {\bf y}_i \}$ 是其对应的数字标签。在该模型设置中，隐变量 ${\bf z}_i$ 和（部分观察到的）数字标签联合生成了观测到的图像。

 ${\bf z}_i$ 代表*除了数字标签之外的所有内容*，例如手写风格、位置等。
 
 先回避一下 $({\bf x}_i, {\bf y}_i, {\bf z}_i)$ 这种分解方式是否合适的问题，重点关注此模型中具有挑战性的推断问题，以及在后文中将介绍的其中一些解决方案。

## 2 推断面临的挑战

为具体起见，我们将继续假设部分缺失的观测 $\{ {\bf y}_i \}$ 是离散标签；同时假设 $\{ {\bf z}_i \}$ 是连续的。

- 如果将随机变分推断的一般性方法应用于我们的模型（参见 [SVI Part I](003_svi_part_i.ipynb)），将需要采样没有被完全观测到的离散型随机变量 ${\bf y}_i$ （因此无法重参数化）。正如 [SVI Part III](005_svi_part_iii.ipynb) 中所讨论的，这会导致高方差的梯度估计。

- 解决该问题的一种常见方法将在下面探讨的：放弃采样，在计算无标签数据点  ${\bf x}_i$ 的 ELBO 时，对类标签 ${\bf y}_i$ 的所有可能的十个值的求和。这每一步成本更高，但可以帮助我们减少梯度估计器的方差，从而减少迭代步骤。

- 回想一下，引导的作用是 `填充` 隐变量。具体来说，引导的一个组分将是一个数字分类器 $q_\phi({\bf y} | {\bf x})$，它将给定图像 $\{ {\bf x}_i \}$ 随机 `填充` 标签值 $\{ {\bf y}_i \}$ 。至关重要的是，这意味着 ELBO 中唯一依赖于 $q_\phi(\cdot | {\bf x})$ 的项是包含无标签数据点求和的项。这意味着分类器 $q_\phi(\cdot | {\bf x})$ —— 在许多情况下是主要感兴趣的对象 —— 不会从有标签数据点中学习（至少不是直接）。

- 这似乎是一个潜在的问题。幸运的是，可以进行各种修复。下面将遵循参考文献 [1] 中的方法，其中包括为分类器引入一个额外的目标函数，以确保分类器直接从有标签数据中学习。

我们已经为我们完成了工作，所以让我们开始吧！

+++

## 3 第一个变体：标准目标函数、朴素估计量

如概述中所讨论，我们考虑 `图 1` 中描述的模型。 更具体的，该`模型`有如下结构：

- $p({\bf y}) = Cat({\bf y}~|~{\bf \pi})$: 类标签的多项（或类别）先验 
- $p({\bf z}) = \mathcal{N}({\bf z}~|~{\bf 0,I})$: 隐编码 $\bf z$ 的单位正态先验
- $p_{\theta}({\bf x}~|~{\bf z,y}) = \text{Bernoulli}\left({\bf x}~|~\mu\left({\bf z,y}\right)\right)$: 参数化的伯努利似然函数; $\mu\left({\bf z,y}\right)$ 对应与代码中的 `decoder` 。

我们构建 `引导` $q_{\phi}(.)$ 的各组分结构如下：

- $q_{\phi}({\bf y}~|~{\bf x}) = Cat({\bf y}~|~{\bf \alpha}_{\phi}\left({\bf x}\right))$: 参数化的多项（或类别）分布; ${\bf \alpha}_{\phi}\left({\bf x}\right)$ 对应于代码中的 `encoder_y` 。
- $q_{\phi}({\bf z}~|~{\bf x, y}) = \mathcal{N}({\bf z}~|~{\bf \mu}_{\phi}\left({\bf x, y}\right), {\bf \sigma^2_{\phi}\left(x, y\right)})$: 参数化的正态分布； ${\bf \mu}_{\phi}\left({\bf x, y}\right)$ 和 ${\bf \sigma^2_{\phi}\left(x, y\right)}$ 对应于代码中的神经网络数字分类器 `encoder_z` 。

+++

这些选择再现了参考文献 [1] 中模型 M2 及其相应推断网络的结构。我们将该`模型/引导对`转换为下面的 Pyro 代码。请注意：

- 标签 `ys` 用独热编码表示，仅部分存在观测值（`None` 表示未观察到的值）。
- `model()` 可以处理有观测值或无观测值的情况 。
- 代码假设 `xs` 和 `ys` 分别是图像和标签的小批量，每批的大小由 `batch_size` 表示。

```{code-cell} ipython3
def model(self, xs, ys=None):
    # register this pytorch module and all of its sub-modules with pyro
    pyro.module("ss_vae", self)
    batch_size = xs.size(0)

    # inform Pyro that the variables in the batch of xs, ys are conditionally independent
    with pyro.plate("data"):

        # sample the handwriting style from the constant prior distribution
        prior_loc = xs.new_zeros([batch_size, self.z_dim])
        prior_scale = xs.new_ones([batch_size, self.z_dim])
        zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

        # if the label y (which digit to write) is supervised, sample from the
        # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
        alpha_prior = xs.new_ones([batch_size, self.output_size]) / (1.0 * self.output_size)
        ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

        # finally, score the image (x) using the handwriting style (z) and
        # the class label y (which digit to write) against the
        # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
        # where `decoder` is a neural network
        loc = self.decoder([zs, ys])
        pyro.sample("x", dist.Bernoulli(loc).to_event(1), obs=xs)

def guide(self, xs, ys=None):
    with pyro.plate("data"):
        # if the class label (the digit) is not supervised, sample
        # (and score) the digit with the variational distribution
        # q(y|x) = categorical(alpha(x))
        if ys is None:
            alpha = self.encoder_y(xs)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha))

        # sample (and score) the latent handwriting-style with the variational
        # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
        loc, scale = self.encoder_z([xs, ys])
        pyro.sample("z", dist.Normal(loc, scale).to_event(1))
```

### 3.1 网络的定义

在我们的实验中，使用与参考文献 [1] 中使用的相同网络配置。编码器和解码器网络各有一个含有 500 个单元和 `softplus` 激活函数的隐藏层：使用 `softmax` 作为 `encoder_y` 的输出激活函数，使用 `sigmoid` 作为 `decoder` 的输出激活函数，并且对 `encoder_z` 输出的`scale` 部分取幂。隐空间维度为 50。

### 3.2 MNIST 的预处理

我们将像素值归一化到 $[0.0, 1.0]$ 范围内。使用来自 `torchvision` 库的 [MNIST 数据加载器](http://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.MNIST)。测试集由 10000 个样本点组成；训练集由 60000 个样本点构成，我们使用前 50000 个样本点训练（分为监督和非监督部分），剩余 10000 个样本点用于验证。

在我们的实验中，训练集将在保证各类别样本数量均衡的情况下，分别采用 $4$ 种有监督配置，即考虑随机选择 $3000$、$1000$、$600$ 和 $100$ 个有监督样本。

+++

### 3.3 目标函数

该模型的目标函数有两项组成 (参考文献 [1] 的公式 8)：

$$\mathcal{J} = \!\!\sum_{({\bf x,y}) \in \mathcal{D}_{supervised} } \!\!\!\!\!\!\!\!\mathcal{L}\big({\bf x,y}\big) +\!\!\! \sum_{{\bf x} \in \mathcal{D}_{unsupervised}} \!\!\!\!\!\!\!\mathcal{U}\left({\bf x}\right)
$$

为了在 Pyro 中实现这一点，我们设置了一个 SVI 类的实例。目标函数中的两个项将自动出现（根据传递的是有标签数据还是无标签数据的 `step` 方法）。我们将交替执行有标签和无标签的小批量步骤，每种类型小批量执行的步骤数取决于有标签数据所占的总比例。例如，如果有 1000 张有标签图像和 49000 张无标签图像，那么对于每个有标签的小批量，我们将使用无标签的小批量执行 49 个步骤。 （注意，可以通过多种不同的方式执行此操作，但为简单起见，我们仅考虑此方法。）

此设置的代码如下：

```{code-cell} ipython3
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam

# setup the optimizer
adam_params = {"lr": 0.0003}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
```

当在 Pyro 中运行此推断时，在测试期间看到的性能因分类变量采样中固有的噪声而降低 (见图 2 和表 1 )。 为了解决这个问题，我们需要一个更好的 ELBO 梯度估计器。

<center><figure>
    <table>
        <tr>
            <td> 
            <img src="http://pyro.ai/examples/_static/img/exp_1_losses_24_3000.png?2" style="width: 450px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/exp_1_acc_24_3000.png?2" style="width: 450px;">
            </td>
        </tr>
    </table> 
    <figcaption> 
        <font size="+1"><b>图 2:</b> 第一种变种</font> <b>(Left)</b> 3000 个有监督样本的损失 <b>(Right)</b> 测试和验证的精度。
    </figcaption>
</figure></center>

## 4 插曲：加和去除隐变量

正如介绍中强调的那样，当没有观测到离散隐标签 ${\bf y}$ 时，ELBO 梯度估计依赖于从 $q_\phi({\bf y}|{\bf x})$ 中采样。这些梯度估计可能具有很高的方差，尤其是在学习过程的早期，当猜测的标签经常不正确时。在此情况下减少方差的常用方法是对离散隐变量求和取代蒙特卡罗期望。

即将蒙特卡洛期望：

$$\mathbb E_{{\bf y}\sim q_\phi(\cdot|{\bf x})}\nabla\operatorname{ELBO}$$

替换为显式的求和：

$$\sum_{\bf y} q_\phi({\bf y}|{\bf x})\nabla\operatorname{ELBO}$$

这个求和通常是手动实现的，如 [1] 中所示，但 Pyro 可以在许多情况下自动执行此操作。为了自动求所有离散隐变量的和（这里只有 ${\bf y}$），我们简单地将`引导`包装在 `config_enumerate()` 中：

```python
svi = SVI(model, config_enumerate(guide), optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))
```

在这种操作模式下，每一个 `svi.step(...)` 为 $y$ 的 10 个隐状态各计算一个梯度项。虽然每一步因此而增加了 $10\times$ 的成本，但它能有效降低梯度估计的方差。

除了本教程中的特定模型外，Pyro 支持对任意多个离散隐变量的求和。请注意，求和的成本随离散变量的数量呈指数增长，但如果将多个独立的离散变量打包到一个张量中（如本教程中所示，其中整个小批量的离散标签被打包），则成本会降至单个张量 ${\bf y}$)。要使用这种并行形式的 `config_enumerate()`，必须通过将向量化代码包装在一个 `with pyro.plate("name")` 代码块中来通知 Pyro ：小批量中的项确实是独立的。

+++

## 5 第二种变体：标准目标函数、更佳的估计器

现在有了离散隐变量求和这个工具，可以看看是否有助于性能。首先，图 3 表明，测试和验证的准确性在训练过程中变得更加平滑。更重要的是，对于 3000 个有标签样本，这种修改将测试准确度从大约 `20%` 提高到大约 `90%` 。完整结果见表 1。这很好，但能做得更好吗？

<center><figure>
    <table>
        <tr>
            <td> 
            <img src="http://pyro.ai/examples/_static/img/exp_2_losses_56_3000.png?2" style="width: 450px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/exp_2_acc_56_3000.png?2" style="width: 450px;">
            </td>
        </tr>
    </table> 
    <figcaption> 
        <font size="+1"><b>图 3:</b> 第二种变种</font> <b>(Left)</b> 3000 个有监督样本的损失 <b>(Right)</b> 测试和验证的精度。
    </figcaption>
</figure></center>


## 6 第三种变体：在目标中增加项

对于我们迄今为止探索的两个变体，分类器 $q_{\phi}({\bf y}~|~ {\bf x})$ 不直接从有标签数据中学习。正如我在概述中所讨论的，这似乎是一个潜在的问题。解决这个问题的一种方法是向目标函数中添加一个额外的项，以便分类器直接从有标签数据中学习。而这正是参考文献 [1] 中采用的方法（参见他们的方程 9）。

修改后的目标函数由下式给出：

$$
\mathcal{J}^{\alpha} = \mathcal{J} + \alpha \mathop{\mathbb{E}}_{\tilde{p_l}({\bf x,y})} \big[-\log\big(q_{\phi}({\bf y}~|~ {\bf x})\big)\big] 
$$

$$
= \mathcal{J} + \alpha' \sum_{({\bf x,y}) \in \mathcal{D}_{\text{supervised}}}  \big[-\log\big(q_{\phi}({\bf y}~|~ {\bf x})\big)\big]
$$


其中 $\tilde{p_l}({\bf x,y})$ 是有标签（或监督）数据和 $\alpha' \equiv \frac{\alpha}{|\mathcal{D}_{\text{supervised}}|}$ 上的经验分布。请注意，此处引入了一个超参数 $\alpha$ 来调节新项的重要性权重。

为了在 Pyro 中使用这个修改后的目标进行学习，执行以下操作： 

- 使用一个新的`模型/引导对`（参见下面的代码片段），对应于给定图像  ${\ bf x}$  时，在有标签观测 ${\bf y}$ 和预测分布 $q_{\phi}({\ bf y}~|~ {\bf x})$ 之间评分。

- 在调用 `pyro.sample` 时，通过使用 `poutine.scale` 来设置尺度因子  $\alpha'$ ；请注意，`poutine.scale` 在 [Deep Markov Model](dmm.ipynb) 中被用于实现 KL 退火的类似效果。

- 创建一个新的 `SVI` 对象并用它对新的目标项采取梯度步骤

```{code-cell} ipython3
def model_classify(self, xs, ys=None):
    pyro.module("ss_vae", self)
    with pyro.plate("data"):
        # this here is the extra term to yield an auxiliary loss
        # that we do gradient descent on
        if ys is not None:
            alpha = self.encoder_y(xs)
            with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

def guide_classify(xs, ys):
    # the guide is trivial, since there are no 
    # latent random variables
    pass

svi_aux = SVI(model_classify, guide_classify, optimizer, loss=Trace_ELBO())
```

当在 Pyro 中使用目标中的附加项运行推断时，性能优于前两种推断设置。例如， $3000$ 个有标签样本的测试准确率从 `90%` 提高到 `96%` （参见图 4 和表 1）。注意，我们使用验证准确性来选择超参数 $\alpha'$ 。

<center><figure>
    <table>
        <tr>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/exp_3_losses_112_3000.png?2"  style="width: 450px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/exp_3_acc_112_3000.png?2" style="width: 450px;">
            </td>
        </tr>
    </table> 
    <figcaption> 
        <font size="+1"><b>图 4:</b> 第三种变体</font> <b>（左）</b> 3000 个有监督样本的训练损失。 <b>（右）</b> 测试和验证精
    </figcaption>
</figure></center>

## 7 结果

### 7.1 各种变体的结果对比

| 有监督数据的数量  | 第一种变体 | 第二种变体 | 第三种变体  | 基线分类器 | 
|------------------|----------------|----------------|----------------|---------------------| 
| 100              | 0.2007(0.0353) | 0.2254(0.0346) | 0.9319(0.0060) | 0.7712(0.0159)      | 
| 600              | 0.1791(0.0244) | 0.6939(0.0345) | 0.9437(0.0070) | 0.8716(0.0064)      | 
| 1000             | 0.2006(0.0295) | 0.7562(0.0235) | 0.9487(0.0038) | 0.8863(0.0025)      | 
| 3000             | 0.1982(0.0522) | 0.8932(0.0159) | 0.9582(0.0012) | 0.9108(0.0015)      | 

<center> <b>表 1:</b> 不同推断方法的结果精度 ( 95% 可信区间) </center>


表 1 汇集了三中变体的结果。为了进行比较，还展示了一个分类器基线的结果，该基线只使用了监督数据（没有隐变量）。表中报告了五种随机比例有监督数据的平均准确度（括号中为 95% 的可信区间）。

我们首先注意到第三种变体的结果再现了参考文献 [1] 中的结果（ 对离散隐变量 $\bf y$ 求和并利用目标函数中的附加项 ）。这是令人鼓舞的，因为这意味着 Pyro 中的抽象被证明足够灵活，可以适应所需的建模和推断设置。值得注意的是，这种灵活性显然是超过基线所必需的。还值得强调的是，我们的生成模型设置的基线和第三种变体之间的差距随着有标签数据点数量的减少而增加（对于只有 100 个有标签数据点的情况，最大约为 15%）。这是一个诱人的结果，因为正是在几乎没有有标签数据点的情况下，半监督学习才显得特别有吸引力。

+++

### 7.2 隐空间的可视化


<center><figure>
    <table>
        <tr>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/third_embedding.png?3" style="width: 450px;">
            </td>
        </tr>
    </table> <center>
    <figcaption> 
        <font size="+1"><b>图 5:</b> 3000 个有监督样本在三种变体中的隐空间嵌入</font> 
    </figcaption> </center>
</figure></center>

我们使用 [T-SNE](https://lvdmaaten.github.io/tsne/) 将隐变量 $\bf z$ 的维数从 $50$ 降维到 $2$ ，并在图 5 中可视化了 10 个数字类。请注意，嵌入的结构与 [VAE](vae.ipynb) 的结构大不相同，其中在嵌入中数字之间分开更明显。这是有道理的，因为对于半监督情况，潜在的 $\bf z$ 可以自由地使用其表示能力来建模，例如手写风格，因为数字之间的变化是由（部分观察到的）标签提供的。

+++

### 7.3 有条件图像生成

我们通过对隐变量 ${\bf z}$ 的不同值进行采样，为每个类别标签（$0$ 到 $9$）采样了 $100$ 个图像。每个数字表现出的手写风格多样性与在 T-SNE 可视化中看到的一致，表明 $\bf z$ 学习到的表示与类标签是分离的。

<center><figure>
    <table>
        <tr>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/0.jpg"  style="width: 200px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/1.jpg" style="width: 200px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/2.jpg"  style="width: 200px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/3.jpg" style="width: 200px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/4.jpg"  style="width: 200px;">
            </td>
        </tr>
        <tr>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/5.jpg"  style="width: 200px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/6.jpg" style="width: 200px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/7.jpg"  style="width: 200px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/8.jpg" style="width: 200px;">
            </td>
            <td> 
                <img src="http://pyro.ai/examples/_static/img/conditional_samples/9.jpg"  style="width: 200px;">
            </td>
        </tr>
    </table> <center>
    <figcaption> 
        <font size="+1"><b>图 6:</b> 通过固定类标签和改变隐变量的方法获得的条件样本（第三种变体 ）</font> 
    </figcaption> </center>
</figure></center>

+++

## 8 最终的思考

我们已经看到生成模型为半监督机器学习提供了一种自然的方法。生成模型最吸引人的特点之一是可以在一个统一设置中探索各种各样的模型。在本教程中，我们只能探索模型和推断设置的一小部分。没有理由预期一种变体是最好的；根据数据集和应用，总有理由更喜欢其中某一个。而且还存在很多可能的变种（见图7）！

<center><figure>
<img src="http://pyro.ai/examples/_static/img/ss_vae_zoo.png" style="width: 300px;">
<figcaption> <center><font size="+1"><b>图 7</b>: 半监督生成模型的家园 </font> </center></figcaption>
</figure></center>

其中一些变体显然比其他变体更有意义，但很难先验地知道哪些变体值得尝试。当我们打开通往更复杂设置的大门时尤其如此，例如图 7 底部的两个模型，除了观测标签缺失之外，还包括一个始终存在的隐变量 $ \tilde{\bf y}$。 顺便说一句，此类模型为我们在上面确定的 `无训练` 问题提供了另一种潜在的解决方案。

如果每个模型和推断过程都是从头开始编写的，那么可能很难说服读者使用。事实上，即使对这些选项中的一小部分进行系统探索也会非常耗时且容易出错。只有通过概率编程系统使模块化和抽象成为可能，我们才能希望以更灵活的方式探索生成模型的前景，并获得可能的预期回报。

本教程完整代码见 [Github](https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/ss_vae_M2.py).

## 参考文献

[1] `Semi-supervised Learning with Deep Generative Models`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling

[2] `Learning Disentangled Representations with Semi-Supervised Deep Generative Models`,
<br/>&nbsp;&nbsp;&nbsp;&nbsp;
N. Siddharth, Brooks Paige, Jan-Willem Van de Meent, Alban Desmaison, Frank Wood, <br/>&nbsp;&nbsp;&nbsp;&nbsp;
Noah D. Goodman, Pushmeet Kohli, Philip H.S. Torr
