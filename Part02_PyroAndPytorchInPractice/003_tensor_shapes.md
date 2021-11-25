---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tensor shapes in Pyro

This tutorial introduces Pyro's organization of tensor dimensions. Before starting, you should familiarize yourself with [PyTorch broadcasting semantics](http://pytorch.org/docs/master/notes/broadcasting.html).  After this tutorial, you may want to also read about [enumeration](http://pyro.ai/examples/enumeration.html).

#### Summary:
- Tensors broadcast by aligning on the right: `torch.ones(3,4,5) + torch.ones(5)`.
- Distribution `.sample().shape == batch_shape + event_shape`.
- Distribution `.log_prob(x).shape == batch_shape` (but not `event_shape`!).
- Use `.expand()` to draw a batch of samples, or rely on `plate` to expand automatically.
- Use `my_dist.to_event(1)` to declare a dimension as dependent.
- Use `with pyro.plate('name', size):` to declare a dimension as conditionally independent.
- All dimensions must be declared either dependent or conditionally independent.
- Try to support batching on the left. This lets Pyro auto-parallelize.
  - use negative indices like `x.sum(-1)` rather than `x.sum(2)`
  - use ellipsis notation like `pixel = image[..., i, j]`
  - use [Vindex](http://docs.pyro.ai/en/dev/ops.html#pyro.ops.indexing.Vindex) if `i,j` are enumerated, `pixel = Vindex(image)[..., i, j]`
- When using `pyro.plate`'s automatic subsampling, be sure to subsample your data:
  - Either manually subample by capturing the index `with pyro.plate(...) as i: ...`
  - or automatically subsample via `batch = pyro.subsample(data, event_dim=...)`.
- When debugging, examine all shapes in a trace using [Trace.format_shapes()](http://docs.pyro.ai/en/dev/poutine.html#pyro.poutine.Trace.format_shapes).
  
#### Table of Contents
- [Distribution shapes](#Distributions-shapes:-batch_shape-and-event_shape)
  - [Examples](#Examples)
  - [Reshaping distributions](#Reshaping-distributions)
  - [It is always safe to assume dependence](#It-is-always-safe-to-assume-dependence)
- [Declaring independence with plate](#Declaring-independent-dims-with-plate)
- [Subsampling inside plate](#Subsampling-tensors-inside-a-plate)
- [Broadcasting to allow Parallel Enumeration](#Broadcasting-to-allow-parallel-enumeration)
  - [Writing parallelizable code](#Writing-parallelizable-code)
  - [Automatic broadcasting inside pyro.plate](#Automatic-broadcasting-inside-pyro-plate)

```{code-cell} ipython3
import os
import torch
import pyro
from torch.distributions import constraints
from pyro.distributions import Bernoulli, Categorical, MultivariateNormal, Normal
from pyro.distributions.util import broadcast_shape
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.poutine as poutine
from pyro.optim import Adam

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.7.0')

# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)
```

## Distributions shapes: `batch_shape` and `event_shape` <a class="anchor" id="Distributions-shapes:-batch_shape-and-event_shape"></a>

PyTorch `Tensor`s have a single `.shape` attribute, but `Distribution`s have two shape attributions with special meaning: `.batch_shape` and `.event_shape`. These two combine to define the total shape of a sample
```py
x = d.sample()
assert x.shape == d.batch_shape + d.event_shape
```
Indices over `.batch_shape` denote conditionally independent random variables, whereas indices over `.event_shape` denote dependent random variables (ie one draw from a distribution). Because the dependent random variables define probability together, the `.log_prob()` method only produces a single number for each event of shape `.event_shape`. Thus the total shape of `.log_prob()` is `.batch_shape`:
```py
assert d.log_prob(x).shape == d.batch_shape
```
Note that the `Distribution.sample()` method also takes a `sample_shape` parameter that indexes over independent identically distributed (iid) random varables, so that
```py
x2 = d.sample(sample_shape)
assert x2.shape == sample_shape + batch_shape + event_shape
```
In summary
```
      |      iid     | independent | dependent
------+--------------+-------------+------------
shape = sample_shape + batch_shape + event_shape
```
For example univariate distributions have empty event shape (because each number is an independent event). Distributions over vectors like `MultivariateNormal` have `len(event_shape) == 1`. Distributions over matrices like `InverseWishart` have `len(event_shape) == 2`.

### Examples <a class="anchor" id="Examples"></a>

The simplest distribution shape is a single univariate distribution.

```{code-cell} ipython3
d = Bernoulli(0.5)
assert d.batch_shape == ()
assert d.event_shape == ()
x = d.sample()
assert x.shape == ()
assert d.log_prob(x).shape == ()
```

Distributions can be batched by passing in batched parameters.

```{code-cell} ipython3
d = Bernoulli(0.5 * torch.ones(3,4))
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)
```

Another way to batch distributions is via the `.expand()` method. This only works if 
parameters are identical along the leftmost dimensions.

```{code-cell} ipython3
d = Bernoulli(torch.tensor([0.1, 0.2, 0.3, 0.4])).expand([3, 4])
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)
```

Multivariate distributions have nonempty `.event_shape`. For these distributions, the shapes of `.sample()` and `.log_prob(x)` differ:

```{code-cell} ipython3
d = MultivariateNormal(torch.zeros(3), torch.eye(3, 3))
assert d.batch_shape == ()
assert d.event_shape == (3,)
x = d.sample()
assert x.shape == (3,)            # == batch_shape + event_shape
assert d.log_prob(x).shape == ()  # == batch_shape
```

### Reshaping distributions <a class="anchor" id="Reshaping-distributions"></a>

In Pyro you can treat a univariate distribution as multivariate by calling the [.to_event(n)](http://docs.pyro.ai/en/dev/distributions.html#pyro.distributions.torch_distribution.TorchDistributionMixin.to_event) property where `n` is the number of batch dimensions (from the right) to declare as *dependent*.

```{code-cell} ipython3
d = Bernoulli(0.5 * torch.ones(3,4)).to_event(1)
assert d.batch_shape == (3,)
assert d.event_shape == (4,)
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3,)
```

While you work with Pyro programs, keep in mind that samples have shape `batch_shape + event_shape`, whereas `.log_prob(x)` values have shape `batch_shape`. You'll need to ensure that `batch_shape` is carefully controlled by either trimming it down with `.to_event(n)` or by declaring dimensions as independent via `pyro.plate`.

### It is always safe to assume dependence <a class="anchor" id="It-is-always-safe-to-assume-dependence"></a>

Often in Pyro we'll declare some dimensions as dependent even though they are in fact independent, e.g.
```py
x = pyro.sample("x", Normal(0, 1).expand([10]).to_event(1))
assert x.shape == (10,)
```
This is useful for two reasons: First it allows us to easily swap in a `MultivariateNormal` distribution later. Second it simplifies the code a bit since we don't need a `plate` (see below) as in
```py
with pyro.plate("x_plate", 10):
    x = pyro.sample("x", Normal(0, 1))  # .expand([10]) is automatic
    assert x.shape == (10,)
```
The difference between these two versions is that the second version with `plate` informs Pyro that it can make use of conditional independence information when estimating gradients, whereas in the first version Pyro must assume they are dependent (even though the normals are in fact conditionally independent). This is analogous to d-separation in graphical models: it is always safe to add edges and assume variables *may* be dependent (i.e. to widen the model class), but it is unsafe to assume independence when variables are actually dependent (i.e. narrowing the model class so the true model lies outside of the class, as in mean field). In practice Pyro's SVI inference algorithm uses reparameterized gradient estimators for `Normal` distributions so both gradient estimators have the same performance.

+++

## Declaring independent dims with `plate` <a class="anchor" id="Declaring-independent-dims-with-plate"></a>

Pyro models can use the context manager [pyro.plate](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate) to declare that certain batch dimensions are independent. Inference algorithms can then take advantage of this independence to e.g. construct lower variance gradient estimators or to enumerate in linear space rather than exponential space. An example of an independent dimension is the index over data in a minibatch: each datum should be independent of all others.

The simplest way to declare a dimension as independent is to declare the rightmost batch dimension as independent via a simple
```py
with pyro.plate("my_plate"):
    # within this context, batch dimension -1 is independent
```
We recommend always providing an optional size argument to aid in debugging shapes
```py
with pyro.plate("my_plate", len(my_data)):
    # within this context, batch dimension -1 is independent
```
Starting with Pyro 0.2 you can additionally nest `plates`, e.g. if you have per-pixel independence:
```py
with pyro.plate("x_axis", 320):
    # within this context, batch dimension -1 is independent
    with pyro.plate("y_axis", 200):
        # within this context, batch dimensions -2 and -1 are independent
```
Note that we always count from the right by using negative indices like -2, -1.

Finally if you want to mix and match `plate`s for e.g. noise that depends only on `x`, some noise that depends only on `y`, and some noise that depends on both, you can declare multiple `plates` and use them as reusable context managers. In this case Pyro cannot automatically allocate a dimension, so you need to provide a `dim` argument (again counting from the right):
```py
x_axis = pyro.plate("x_axis", 3, dim=-2)
y_axis = pyro.plate("y_axis", 2, dim=-3)
with x_axis:
    # within this context, batch dimension -2 is independent
with y_axis:
    # within this context, batch dimension -3 is independent
with x_axis, y_axis:
    # within this context, batch dimensions -3 and -2 are independent
```
Let's take a closer look at batch sizes within `plate`s.

```{code-cell} ipython3
def model1():
    a = pyro.sample("a", Normal(0, 1))
    b = pyro.sample("b", Normal(torch.zeros(2), 1).to_event(1))
    with pyro.plate("c_plate", 2):
        c = pyro.sample("c", Normal(torch.zeros(2), 1))
    with pyro.plate("d_plate", 3):
        d = pyro.sample("d", Normal(torch.zeros(3,4,5), 1).to_event(2))
    assert a.shape == ()       # batch_shape == ()     event_shape == ()
    assert b.shape == (2,)     # batch_shape == ()     event_shape == (2,)
    assert c.shape == (2,)     # batch_shape == (2,)   event_shape == ()
    assert d.shape == (3,4,5)  # batch_shape == (3,)   event_shape == (4,5) 

    x_axis = pyro.plate("x_axis", 3, dim=-2)
    y_axis = pyro.plate("y_axis", 2, dim=-3)
    with x_axis:
        x = pyro.sample("x", Normal(0, 1))
    with y_axis:
        y = pyro.sample("y", Normal(0, 1))
    with x_axis, y_axis:
        xy = pyro.sample("xy", Normal(0, 1))
        z = pyro.sample("z", Normal(0, 1).expand([5]).to_event(1))
    assert x.shape == (3, 1)        # batch_shape == (3,1)     event_shape == ()
    assert y.shape == (2, 1, 1)     # batch_shape == (2,1,1)   event_shape == ()
    assert xy.shape == (2, 3, 1)    # batch_shape == (2,3,1)   event_shape == ()
    assert z.shape == (2, 3, 1, 5)  # batch_shape == (2,3,1)   event_shape == (5,)
    
test_model(model1, model1, Trace_ELBO())
```

It is helpful to visualize the `.shape`s of each sample site by aligning them at the boundary between `batch_shape` and `event_shape`: dimensions to the right will be summed out in `.log_prob()` and dimensions to the left will remain. 
```
batch dims | event dims
-----------+-----------
           |        a = sample("a", Normal(0, 1))
           |2       b = sample("b", Normal(zeros(2), 1)
           |                        .to_event(1))
           |        with plate("c", 2):
          2|            c = sample("c", Normal(zeros(2), 1))
           |        with plate("d", 3):
          3|4 5         d = sample("d", Normal(zeros(3,4,5), 1)
           |                       .to_event(2))
           |
           |        x_axis = plate("x", 3, dim=-2)
           |        y_axis = plate("y", 2, dim=-3)
           |        with x_axis:
        3 1|            x = sample("x", Normal(0, 1))
           |        with y_axis:
      2 1 1|            y = sample("y", Normal(0, 1))
           |        with x_axis, y_axis:
      2 3 1|            xy = sample("xy", Normal(0, 1))
      2 3 1|5           z = sample("z", Normal(0, 1).expand([5])
           |                       .to_event(1))
```
To examine the shapes of sample sites in a program automatically, you can trace the program and use the [Trace.format_shapes()](http://docs.pyro.ai/en/dev/poutine.html#pyro.poutine.Trace.format_shapes) method, which prints three shapes for each sample site: the distribution shape (both `site["fn"].batch_shape` and `site["fn"].event_shape`), the value shape (`site["value"].shape`), and if log probability has been computed also the `log_prob` shape (`site["log_prob"].shape`):

```{code-cell} ipython3
trace = poutine.trace(model1).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())
```

## Subsampling tensors inside a `plate` <a class="anchor" id="Subsampling-tensors-inside-a-plate"></a>

One of the main uses of [plate](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate) is to subsample data. This is possible within a `plate` because data are conditionally independent, so the expected value of the loss on, say, half the data should be half the expected loss on the full data.

To subsample data, you need to inform Pyro of both the original data size and the subsample size; Pyro will then choose a random subset of data and yield the set of indices.

```{code-cell} ipython3
data = torch.arange(100.)

def model2():
    mean = pyro.param("mean", torch.zeros(len(data)))
    with pyro.plate("data", len(data), subsample_size=10) as ind:
        assert len(ind) == 10    # ind is a LongTensor that indexes the subsample.
        batch = data[ind]        # Select a minibatch of data.
        mean_batch = mean[ind]   # Take care to select the relevant per-datum parameters.
        # Do stuff with batch:
        x = pyro.sample("x", Normal(mean_batch, 1), obs=batch)
        assert len(x) == 10
        
test_model(model2, guide=lambda: None, loss=Trace_ELBO())
```

## Broadcasting to allow parallel enumeration <a class="anchor" id="Broadcasting-to-allow-parallel-enumeration"></a>

Pyro 0.2 introduces the ability to enumerate discrete latent variables in parallel. This can significantly reduce the variance of gradient estimators when learning a posterior via [SVI](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.svi.SVI).

To use parallel enumeration, Pyro needs to allocate tensor dimension that it can use for enumeration. To avoid conflicting with other dimensions that we want to use for `plate`s, we need to declare a budget of the maximum number of tensor dimensions we'll use. This budget is called `max_plate_nesting` and is an argument to [SVI](http://docs.pyro.ai/en/dev/inference_algos.html) (the argument is simply passed through to [TraceEnum_ELBO](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.traceenum_elbo.TraceEnum_ELBO)). Usually Pyro can determine this budget on its own (it runs the `(model,guide)` pair once and record what happens), but in case of dynamic model structure you may need to declare `max_plate_nesting` manually.

To understand `max_plate_nesting` and how Pyro allocates dimensions for enumeration, let's revisit `model1()` from above. This time we'll map out three types of dimensions:
enumeration dimensions on the left (Pyro takes control of these), batch dimensions in the middle, and event dimensions on the right.

+++

```
      max_plate_nesting = 3
           |<--->|
enumeration|batch|event
-----------+-----+-----
           |. . .|      a = sample("a", Normal(0, 1))
           |. . .|2     b = sample("b", Normal(zeros(2), 1)
           |     |                      .to_event(1))
           |     |      with plate("c", 2):
           |. . 2|          c = sample("c", Normal(zeros(2), 1))
           |     |      with plate("d", 3):
           |. . 3|4 5       d = sample("d", Normal(zeros(3,4,5), 1)
           |     |                     .to_event(2))
           |     |
           |     |      x_axis = plate("x", 3, dim=-2)
           |     |      y_axis = plate("y", 2, dim=-3)
           |     |      with x_axis:
           |. 3 1|          x = sample("x", Normal(0, 1))
           |     |      with y_axis:
           |2 1 1|          y = sample("y", Normal(0, 1))
           |     |      with x_axis, y_axis:
           |2 3 1|          xy = sample("xy", Normal(0, 1))
           |2 3 1|5         z = sample("z", Normal(0, 1).expand([5]))
           |     |                     .to_event(1))
```
Note that it is safe to overprovision `max_plate_nesting=4` but we cannot underprovision `max_plate_nesting=2` (or Pyro will error). Let's see how this works in practice.

```{code-cell} ipython3
@config_enumerate
def model3():
    p = pyro.param("p", torch.arange(6.) / 6)
    locs = pyro.param("locs", torch.tensor([-1., 1.]))

    a = pyro.sample("a", Categorical(torch.ones(6) / 6))
    b = pyro.sample("b", Bernoulli(p[a]))  # Note this depends on a.
    with pyro.plate("c_plate", 4):
        c = pyro.sample("c", Bernoulli(0.3))
        with pyro.plate("d_plate", 5):
            d = pyro.sample("d", Bernoulli(0.4))
            e_loc = locs[d.long()].unsqueeze(-1)
            e_scale = torch.arange(1., 8.)
            e = pyro.sample("e", Normal(e_loc, e_scale)
                            .to_event(1))  # Note this depends on d.

    #                   enumerated|batch|event dims
    assert a.shape == (         6, 1, 1   )  # Six enumerated values of the Categorical.
    assert b.shape == (      2, 1, 1, 1   )  # Two enumerated Bernoullis, unexpanded.
    assert c.shape == (   2, 1, 1, 1, 1   )  # Only two Bernoullis, unexpanded.
    assert d.shape == (2, 1, 1, 1, 1, 1   )  # Only two Bernoullis, unexpanded.
    assert e.shape == (2, 1, 1, 1, 5, 4, 7)  # This is sampled and depends on d.

    assert e_loc.shape   == (2, 1, 1, 1, 1, 1, 1,)
    assert e_scale.shape == (                  7,)
            
test_model(model3, model3, TraceEnum_ELBO(max_plate_nesting=2))
```

Let's take a closer look at those dimensions. First note that Pyro allocates enumeration dims starting from the right at `max_plate_nesting`: Pyro allocates dim -3 to enumerate `a`, then dim -4 to enumerate `b`, then dim -5 to enumerate `c`, and finally dim -6 to enumerate `d`. Next note that samples only have extent (size > 1) in the new enumeration dimension. This helps keep tensors small and computation cheap. (Note that the `log_prob` shape will be broadcast up to contain both enumeratin shape and batch shape, so e.g. `trace.nodes['d']['log_prob'].shape == (2, 1, 1, 1, 5, 4)`.)

We can draw a similar map of the tensor dimensions:
```
     max_plate_nesting = 2
            |<->|
enumeration batch event
------------|---|-----
           6|1 1|     a = pyro.sample("a", Categorical(torch.ones(6) / 6))
         2 1|1 1|     b = pyro.sample("b", Bernoulli(p[a]))
            |   |     with pyro.plate("c_plate", 4):
       2 1 1|1 1|         c = pyro.sample("c", Bernoulli(0.3))
            |   |         with pyro.plate("d_plate", 5):
     2 1 1 1|1 1|             d = pyro.sample("d", Bernoulli(0.4))
     2 1 1 1|1 1|1            e_loc = locs[d.long()].unsqueeze(-1)
            |   |7            e_scale = torch.arange(1., 8.)
     2 1 1 1|5 4|7            e = pyro.sample("e", Normal(e_loc, e_scale)
            |   |                             .to_event(1))
```
To automatically examine this model with enumeration semantics, we can create an enumerated trace and then use [Trace.format_shapes()](http://docs.pyro.ai/en/dev/poutine.html#pyro.poutine.Trace.format_shapes):

```{code-cell} ipython3
trace = poutine.trace(poutine.enum(model3, first_available_dim=-3)).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())
```

### Writing parallelizable code <a class="anchor" id="Writing-parallelizable-code"></a>

It can be tricky to write Pyro models that correctly handle parallelized sample sites. Two tricks help: [broadcasting](http://pytorch.org/docs/master/notes/broadcasting.html) and [ellipsis slicing](http://python-reference.readthedocs.io/en/dev/docs/brackets/ellipsis.html). Let's look at a contrived model to see how these work in practice. Our aim is to write a model that works both with and without enumeration.

```{code-cell} ipython3
width = 8
height = 10
sparse_pixels = torch.LongTensor([[3, 2], [3, 5], [3, 9], [7, 1]])
enumerated = None  # set to either True or False below

def fun(observe):
    p_x = pyro.param("p_x", torch.tensor(0.1), constraint=constraints.unit_interval)
    p_y = pyro.param("p_y", torch.tensor(0.1), constraint=constraints.unit_interval)
    x_axis = pyro.plate('x_axis', width, dim=-2)
    y_axis = pyro.plate('y_axis', height, dim=-1)

    # Note that the shapes of these sites depend on whether Pyro is enumerating.
    with x_axis:
        x_active = pyro.sample("x_active", Bernoulli(p_x))
    with y_axis:
        y_active = pyro.sample("y_active", Bernoulli(p_y))
    if enumerated:
        assert x_active.shape  == (2, 1, 1)
        assert y_active.shape  == (2, 1, 1, 1)
    else:
        assert x_active.shape  == (width, 1)
        assert y_active.shape  == (height,)

    # The first trick is to broadcast. This works with or without enumeration.
    p = 0.1 + 0.5 * x_active * y_active
    if enumerated:
        assert p.shape == (2, 2, 1, 1)
    else:
        assert p.shape == (width, height)
    dense_pixels = p.new_zeros(broadcast_shape(p.shape, (width, height)))

    # The second trick is to index using ellipsis slicing.
    # This allows Pyro to add arbitrary dimensions on the left.
    for x, y in sparse_pixels:
        dense_pixels[..., x, y] = 1
    if enumerated:
        assert dense_pixels.shape == (2, 2, width, height)
    else:
        assert dense_pixels.shape == (width, height)

    with x_axis, y_axis:    
        if observe:
            pyro.sample("pixels", Bernoulli(p), obs=dense_pixels)

def model4():
    fun(observe=True)

def guide4():
    fun(observe=False)

# Test without enumeration.
enumerated = False
test_model(model4, guide4, Trace_ELBO())

# Test with enumeration.
enumerated = True
test_model(model4, config_enumerate(guide4, "parallel"),
           TraceEnum_ELBO(max_plate_nesting=2))
```

### Automatic broadcasting inside pyro.plate<a class="anchor" id="Automatic-broadcasting-inside-pyro-plate"></a>

Note that in all our model/guide specifications, we have relied on [pyro.plate](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate) to automatically expand sample shapes to satisfy the constraints on batch shape enforced by `pyro.sample` statements. However this broadcasting is equivalent to hand-annotated `.expand()` statements.

We will demonstrate this using `model4` from the [previous section](#Writing-parallelizable-code). Note the following changes to the code from earlier:

 - For the purpose of this example, we will only consider "parallel" enumeration, but broadcasting should work as expected without enumeration or with "sequential" enumeration.
 - We have separated out the sampling function which returns the tensors corresponding to the active pixels. Modularizing the model code into components is a common practice, and helps with maintainability of large models.
 - We would also like to use the `pyro.plate` construct to parallelize the ELBO estimator over [num_particles](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.elbo.ELBO). This is done by wrapping the contents of model/guide inside an outermost `pyro.plate` context.

```{code-cell} ipython3
num_particles = 100  # Number of samples for the ELBO estimator
width = 8
height = 10
sparse_pixels = torch.LongTensor([[3, 2], [3, 5], [3, 9], [7, 1]])

def sample_pixel_locations_no_broadcasting(p_x, p_y, x_axis, y_axis):
    with x_axis:
        x_active = pyro.sample("x_active", Bernoulli(p_x).expand([num_particles, width, 1]))
    with y_axis:
        y_active = pyro.sample("y_active", Bernoulli(p_y).expand([num_particles, 1, height]))
    return x_active, y_active

def sample_pixel_locations_full_broadcasting(p_x, p_y, x_axis, y_axis):
    with x_axis:
        x_active = pyro.sample("x_active", Bernoulli(p_x))
    with y_axis:
        y_active = pyro.sample("y_active", Bernoulli(p_y))
    return x_active, y_active 

def sample_pixel_locations_partial_broadcasting(p_x, p_y, x_axis, y_axis):
    with x_axis:
        x_active = pyro.sample("x_active", Bernoulli(p_x).expand([width, 1]))
    with y_axis:
        y_active = pyro.sample("y_active", Bernoulli(p_y).expand([height]))
    return x_active, y_active 

def fun(observe, sample_fn):
    p_x = pyro.param("p_x", torch.tensor(0.1), constraint=constraints.unit_interval)
    p_y = pyro.param("p_y", torch.tensor(0.1), constraint=constraints.unit_interval)
    x_axis = pyro.plate('x_axis', width, dim=-2)
    y_axis = pyro.plate('y_axis', height, dim=-1)

    with pyro.plate("num_particles", 100, dim=-3):
        x_active, y_active = sample_fn(p_x, p_y, x_axis, y_axis)
        # Indices corresponding to "parallel" enumeration are appended 
        # to the left of the "num_particles" plate dim.
        assert x_active.shape  == (2, 1, 1, 1)
        assert y_active.shape  == (2, 1, 1, 1, 1)
        p = 0.1 + 0.5 * x_active * y_active
        assert p.shape == (2, 2, 1, 1, 1)

        dense_pixels = p.new_zeros(broadcast_shape(p.shape, (width, height)))
        for x, y in sparse_pixels:
            dense_pixels[..., x, y] = 1
        assert dense_pixels.shape == (2, 2, 1, width, height)

        with x_axis, y_axis:    
            if observe:
                pyro.sample("pixels", Bernoulli(p), obs=dense_pixels)

def test_model_with_sample_fn(sample_fn):
    def model():
        fun(observe=True, sample_fn=sample_fn)

    @config_enumerate
    def guide():
        fun(observe=False, sample_fn=sample_fn)

    test_model(model, guide, TraceEnum_ELBO(max_plate_nesting=3))

test_model_with_sample_fn(sample_pixel_locations_no_broadcasting)
test_model_with_sample_fn(sample_pixel_locations_full_broadcasting)
test_model_with_sample_fn(sample_pixel_locations_partial_broadcasting)
```

In the first sampling function, we had to do some manual book-keeping and expand the `Bernoulli` distribution's batch shape to account for the conditionally independent dimensions added by the `pyro.plate` contexts. In particular, note how `sample_pixel_locations` needs knowledge of `num_particles`, `width` and `height` and is accessing these variables from the global scope, which is not ideal. 

 - The second argument to `pyro.plate`, i.e. the optional `size` argument needs to be provided for implicit broadasting, so that it can infer the batch shape requirement for each of the sample sites. 
 - The existing `batch_shape` of the sample site must be broadcastable with the size of the `pyro.plate` contexts. In our particular example, `Bernoulli(p_x)` has an empty batch shape which is universally broadcastable.

Note how simple it is to achieve parallelization via tensorized operations using `pyro.plate`! `pyro.plate` also helps in code modularization because model components can be written agnostic of the `plate` contexts in which they may subsequently get embedded in.

```{code-cell} ipython3

```
