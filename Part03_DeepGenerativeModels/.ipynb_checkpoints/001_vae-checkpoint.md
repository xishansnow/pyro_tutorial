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

# 变分自编码器

## 1 概述

变分自编码器 (VAE) 可以说是实现深度概率建模的一种最简单的设置。请注意，我们在这里选择语言时非常小心，并没有使用 “模型” 这个词。 因为变分自编码器本身并不是一个模型，而是用于对某一类特殊模型进行变分推断的特定设置。此类模型非常常见，基本上具有隐变量的任何（无监督）密度估计器都符合此类模型。这些模型的基本结构很简单（见图 1）。

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211129174309-c9ac.webp" alt="fig1" style="zoom: 33%;" />

> **图 1：** 我们感兴趣的深度模型类型

图 1 将我们感兴趣的这类模型的结构描述为概率图形式。有 $N$ 个观测数据点 $\{ \bf x_i \}$。每个数据点由局部隐变量 $\bf z_i$ 生成；还有一个全局参数 $\theta$，因为所有数据点都依赖于它（被绘制在矩形之外的原因）。注意 $\theta$ 是一个参数，而不是随机变量。特别重要的是，我们允许观测变量 $\bf x_i$ 以复杂的非线性方式依赖于局部隐变量 $\bf z_i$。在实践中，这种依赖关系被一个权重参数为 $\theta$ 的（深度）神经网络参数化，其非线性使此类模型的推断特别具有挑战性。

当然，这种非线性结构也为复杂数据建模提供了一种非常灵活的方法。值得强调的是，模型的每个组件都能够以不同方式“重新配置”。例如：

- $p_\theta({\bf x} | {\bf z})$ 中的神经网络可以根据层数、非线性类型、隐藏单元的数量等而变化

- 我们可以选择适合手头数据集的观测似然，如：高斯、伯努利、分类等 

- 我们可以选择隐空间的维数

概率图模型表示是一种考虑模型结构的有效方法，但查看联合概率密度的显式分解也很有成效：

$$
p({\bf x}, {\bf z}) = \prod_{i=1}^N p_\theta({\bf x}_i | {\bf z}_i) p({\bf z}_i)  
$$

$p({\bf x}, {\bf z})$ 分解为若干项的乘积，因此我们将 $\bf z_i$ 称为局部随机变量具有非常清晰的语义。对于任何特定的 $i$，只有单个数据点 $\bf x_i$ 依赖于 $\bf z_i$。因此 $\{\bf z_i\}$ 描述了局部结构，即每个数据点私有的结构。这种分解结构也意味着，我们可以在学习过程中进行二次抽样。因此，这种模型适用于大数据的应用场景。 

> 注：
> 有关此主题和相关主题的更多讨论，请参见 [随机变分推断 第二部分](004_svi_part_ii.ipynb)。

回想一下，引导的工作是 “猜测” 隐变量的“良好”值 —— 良好的意义在于它对模型先验和数据都是真实的。如果不使用摊销，我们将为每个数据点 $\bf x_i$ 引入变分参数 $\{ \lambda_i \}$。这些变分参数代表了我们对 $\bf z_i$ 是 “良好” 值的信念，常见的变分参数设置如：隐变量  ${\bf z}_i$  呈高斯分布时的均值参数和方差参数。

而摊销意味着：我们不为每个数据点引入变分参数 $\{ \lambda_i \}$，而是学习一个能够将每个数据点 $\bf x_i$ 映射到适当 $\lambda_i$ 的函数。我们需要这个函数足够灵活，因此将其参数化为一个神经网络。由此我们最终得到了隐变量 $\bf z$ 空间上的参数化分布族，能够被所有 $N$ 个数据点 ${\bf x}_i$ 实例化（见图 2）。

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211129174302-7d73.webp" style="zoom: 33%;" />

> **图 2：** 引导的概率图表示。

请注意，引导 $q_{\phi}({\bf z} | {\bf x})$ 被（所有数据点共享的）全局参数 $\phi$ 参数化。推断的目标是找到 $\theta$ 和 $\phi$ 的 “良好” 值，以便满足两个条件：

- 对数证据 $\log p_\theta({\bf x})$ 很大，这意味着模型能够很好地解释数据 。
- 引导 $q_{\phi}({\bf z} | {\bf x})$ 提供了对后验的良好近似。

（有关随机变分推断的介绍，请参阅 [SVI Part I](003_svi_part_i.ipynb)。）

为了具体起见，假设 $\{ \bf x_i \}$ 是图像，因此该模型为图像生成模型。一旦我们了解了 $\theta$ 的 “良好” 值，就可以从模型中按照如下流程生成图像：

- 根据先验分布 $p({\bf z})$ 采样得到 $\bf z$ 。

- 根据采样得到的 $\bf z$ 值，从似然 $p_\theta({\bf x}|{\bf z})$ 中采样得到 $\bf x $ 。

每个图像都由一个隐编码 $\bf z$ 表示，该编码通过似然被映射到图像，而似然则取决学到的 $\theta$ 。这就是为什么在此情况下似然通常被称为解码器：它的工作是将 $\bf z$ 解码为 $\bf x$。

注意，这是一个概率模型，因此给定数据点 $\bf x$ 的隐编码 $\bf z$ 存在不确定性。

一旦我们确定了 $\theta$ 和 $\phi$ 的 “良好” 值，则能够进行以下练习：

- 从给定的图像 $\bf x$ 开始；

- 使用引导将其编码为 $\bf z$ ；

- 使用 $\bf z$ 解码模型似然，并获得重建图像 ${\bf x}_{ \rm reco}$ 。

如果我们已经学习了 $\theta$ 和 $\phi$ 的 “良好” 值，则 $\bf x$ 和 ${\bf x}_{\rm reco}$ 应该是相似的。这阐明了 “自动编码器” 这个词是如何被用来描述该设置的：模型是解码器，引导是编码器。它们一起被认为是一个自动编码器。

## 2 Pyro 中的变分自编码器

让我们看看如何在 Pyro 中实现 VAE。要建模的数据集是 MNIST，手写数字图像的集合。由于这是一个流行的基准数据集，我们可以利用 PyTorch 方便的数据加载器来减少需要编写的代码数量：

```{code-cell} ipython3
import os

import numpy as np
import torch
from pyro.contrib.examples.util import MNIST
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
```

```{code-cell} ipython3
assert pyro.__version__.startswith('1.7.0')
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ  
```

```{code-cell} ipython3
# for loading and batching MNIST dataset
def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = MNIST(root=root, train=True, transform=trans,
                      download=download)
    test_set = MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
```

> **注意：**
> 这里需要注意的是使用 `transforms.ToTensor()` 将像素强度归一化了到 $[0.0, 1.0]$ 范围内。

接下来定义一个 PyTorch 模块来封装解码器网络：

```{code-cell} ipython3
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img
```

给定隐编码 $z$ ，`Decoder` 的前向调用将返回`图像空间`中伯努利分布的参数（二值图的原因）。由于每张图像的大小为 $28\times 28=784$，`loc_img` 的大小为 `batch_size` x 784。

接下来定义一个 PyTorch 模块来封装我们的编码器网络：

```{code-cell} ipython3
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale
```

给定图像 $\bf x$，`Encoder` 的前向调用将返回一个均值参数和一个协方差参数，它们共同参数化了隐空间中的（对角）高斯分布。

有了编码器和解码器网络，现在可以写下随机函数来表示我们的`模型`和`引导`。先上模型：

```{code-cell} ipython3
# define the model p(x|z)p(z)
def model(self, x):
    # register PyTorch module `decoder` with Pyro
    pyro.module("decoder", self.decoder)
    with pyro.plate("data", x.shape[0]):
        # setup hyperparameters for prior p(z)
        z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
        z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        # decode the latent code z
        loc_img = self.decoder(z)
        # score against actual images
        pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))
```

> **注意：**
> `model()` 是一个可调用的对象，它将小批量图像 `x` 作为输入。这是一个大小为 $batch\_size \times 784$ 的 `torch.Tensor`。

我们在 `model()` 中做的第一件事是使用 Pyro 注册（之前实例化的）解码器模块。请注意，我们为其提供了一个适当且唯一的名称。对 `pyro.module` 的调用让 Pyro 能够知道解码器网络内部的所有参数。

接下来为先验设置超参数，它是一个单位正态高斯分布。请注意：

- 我们通过`pyro.plate`特别指定了小批量（即最左边的维度）中的数据之间存在独立性。另外，在从隐变量 `z` 中采样时，使用了 `.to_event(1)` 。这确保我们将数据视为具有对角协方差的多元正态分布的样本，而不是 `batch_size = z_dim` 的一元正态分布的样本。因此，当我们为 “隐” 样本评估 `.log_prob` 时，会对每个维度的对数概率求和。更多详细信息，请参阅 [Tensor Shapes](tensor_shapes.ipynb) 教程。

- 由于我们正在处理整个 `mini-batch` 图像，我们需要 `z_loc` 和 `z_scale` 最左边的维度等于 `mini-batch` 大小。

- 如果在 GPU 上，使用 `new_zeros` 和 `new_ones` 来确保新创建的张量在同一个 GPU 设备上。

接下来，我们从先验中采样隐变量 `z`，确保为随机变量提供一个唯一的 Pyro 名称 `'latent'`。

然后通过解码器网络传递 `z`，它返回 `loc_img`。然后，我们在由 `loc_img` 参数化的伯努利似然和小批量 `x` 中的观测图像之间做比较。请注意，此处将展平了 `x` ，以便所有像素都在最右边维度上。

> **注意：** 
> `model()` 中 Pyro 元语流与模型生成过程之间的紧密程度（如图 1 所示）。

现在转到 `引导`：

```{code-cell} ipython3
# define the guide (i.e. variational distribution) q(z|x)
def guide(self, x):
    # register PyTorch module `encoder` with Pyro
    pyro.module("encoder", self.encoder)
    with pyro.plate("data", x.shape[0]):
        # use the encoder to get the parameters used to define q(z|x)
        z_loc, z_scale = self.encoder(x)
        # sample the latent code z
        pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
```

就像模型一样，首先向 Pyro 注册正在使用的 PyTorch 模块（即 `编码器`）。我们采用小批量图像 `x` 并将其传递给编码器。然后使用编码器网络输出的参数，我们使用正态分布对 `mini-batch` 中的每个图像的隐编码进行采样。至关重要的是，我们对隐变量使用了与模型中相同的名称：`'latent'` 。另外，注意使用 `pyro.plate` 来指定 `mini-batch` 维度的独立性，以及使用 `.to_event(1)` 来强制依赖于 `z_dims`，就像在模型中所做的那样。

现在已经定义了完整的`模型`和`引导`，后面可以进行推理了。但做推断之前，先看一下如何将`模型`和`引导`打包到 PyTorch 模块中：

```{code-cell} ipython3
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
```

> **说明：**
> `Module` 的 `encoder` 和 `decoder` 是 `VAE` 模块的属性，这导致它们都被自动注册为隶属于 `VAE` 模块。因此，当对 `VAE` 的某个实例调用 `parameters()` 时，PyTorch 会返回所有的相关参数。这同时也意味着，如果要在 GPU 上运行，对 `cuda()` 的调用将会把所有（子）模块的所有参数均移到 GPU 内存中。

+++

## 3 变分自编码器的推断

现在可以进行推理了。请参阅下一节中的完整代码。首先创建一个 `VAE` 模块的实例。

```{code-cell} ipython3
vae = VAE()
```

然后设置一个 Adam 优化器。

```{code-cell} ipython3
optimizer = Adam({"lr": 1.0e-3})
```

然后设置我们的推断算法，该算法将通过最大化 ELBO 来学习`模型`和`引导`的 “良好” 参数：

```{code-cell} ipython3
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
```

现在只需要定义我们的训练循环：

```{code-cell} ipython3
def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, _ in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train
```

> **注意：**
> 所有小批量逻辑都由数据加载器来处理。训练循环的核心是`svi.step(x)`。这里有两件事我们应该提请注意：
> -`step` 的任何参数都会传递给`模型`和`引导`，因此两者必须具有相同的接口形式
> -`step` 返回损失的含噪声估计。这个估计没有以任何方式做归一化，所以会随着批量大小而缩放

评估逻辑的编码形式类似：

```{code-cell} ipython3
def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x, _ in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test
```

基本上需要做的唯一改变是调用 `evaluate_loss` 而不是 `step`。该函数将计算 ELBO 的估计，但不会采取任何梯度步骤。

需要强调 `VAE` 类的最后一段代码中有一个辅助方法 `reconstruct_img()`。这只是将概述中描述的图像重建实验翻译成了代码。我们取一张图像并将其通过编码器，然后使用编码器提供的高斯分布在隐空间中进行采样得到隐编码。最后，将隐编码解码为图像：我们返回均值向量 `loc_img`，而不是对其进行采样。

> 注意：
> 由于`sample()` 语句是随机的，每次运行 `reconstruct_img` 时，都会抽取得到不同的 `z` 。如果已经学习了一个好的`模型`和`引导` —— 特别是如果我们已经学习了一个好的隐表示——  `z` 样本的多样性将对应于不同的数字书写风格，而且重建的图像应该表现出不同风格。

+++

## 4 代码和示例结果

训练对应于最大化训练数据集的证据下限 (ELBO)。我们训练 100 次迭代并评估测试数据集的 ELBO，见图 3。

```{code-cell} ipython3
# Run options
LEARNING_RATE = 1.0e-3
USE_CUDA = False

# Run only for a single iteration for testing
NUM_EPOCHS = 1 if smoke_test else 100
TEST_FREQUENCY = 5
```

```{code-cell} ipython3
train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=USE_CUDA)

# clear param store
pyro.clear_param_store()

# setup the VAE
vae = VAE(use_cuda=USE_CUDA)

# setup the optimizer
adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)

# setup the inference algorithm
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

train_elbo = []
test_elbo = []
# training loop
for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
```

![](http://pyro.ai/examples/_static/img/vae_plots/test_elbo_vae.png)
> **图 3：** 在训练过程中测试 ELBO 如何发展 


接下来展示一组从模型中随机采样的图像，这些图像都是通过抽取 `z` 的随机样本并为其生成一个图像而得到的，参见图 4。

![](http://pyro.ai/examples/_static/img/vae_plots/vae_embeddings_pt1.jpg)
![](http://pyro.ai/examples/_static/img/vae_plots/vae_embeddings_pt2.jpg)

> **图 4：** 来自生成模型的样本

我们通过编码所有 MNIST 图像来研究整个测试数据集的 50 维潜在空间，并将其均值嵌入二维 T-SNE 空间。然后按类别为每个嵌入的图像着色。结果如图 5 所示，显示了多个类别之间的划分，以及每个类别内部的方差。

![](http://pyro.ai/examples/_static/img/vae_plots/VAE_embedding.png)

> **图 5：** 隐变量 $z$ 的 T-SNE 嵌入。不同颜色对应不同的数字类别。

完整代码参见 [Github](https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py).

## 参考文献

[1] `Auto-Encoding Variational Bayes`, Diederik P Kingma, Max Welling

[2] `Stochastic Backpropagation and Approximate Inference in Deep Generative Models`, Danilo Jimenez Rezende, Shakir Mohamed, Daan Wierstra
