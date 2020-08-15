# 神经网络（Neural Network）
不管具体模型是什么，参数的更新方式都是一样的：反向传播误差然后通过计算损失关于参数的梯度来更新这些参数。

![](https://tangshusen.me/Deep-Learning-with-PyTorch-Chinese/img/chapter5/5.1.png)

## 神经元
深度学习的核心是神经网络，即能够通过简单函数的组合来表示复杂函数的数学实体。

从本质上讲，神经元不过是输入的线性变换（例如，输入乘以一个数[weight，权重]，再加上一个常数[偏置，bias]），然后再经过一个固定的非线性函数（称为激活函数）。
![](https://tangshusen.me/Deep-Learning-with-PyTorch-Chinese/img/chapter5/5.2.png)
**激活函数**
* 是非线性的。在没有激活函数的情况下重复应用$wx+bw x + bwx+b$会产生多项式。非线性的激活函数允许整个网络能近似更复杂的函数。
* 是可微的。激活函数是可微的这样就可以计算穿过它们的梯度。不可微的离散点是无伤大雅的，例如<code>Hardtanh</code>和<code>ReLU</code>。

激活函数还通常（尽管并非总是如此）

* 具有至少一个敏感范围，其中输入的轻微变化会导致输出中相应的变化。
* 具有至少一个不敏感（或饱和）范围，其中输入的变化导致输出的变化很小甚至没有变化。


通常（但并非普遍如此），激活函数至少具有以下特点之一：

* 当输入变为负无穷大时接近（或达到）下限
* 当输入变为正无穷大时接近（或达到）上限

## PyTorch的nn模块
<p style="user-select: auto;">PyTorch有一个专门用于神经网络的完整子模块：<code style="user-select: auto;">torch.nn</code>。该子模块包含创建各种神经网络体系结构所需的构建块。这些构建块在PyTorch术语中称为module（模块），在其他框架中称为layer（层）。</p>
<p style="user-select: auto;">PyTorch模块都是从基类<code style="user-select: auto;">nn.Module</code>继承而来的Python类。模块可以具有一个或多个参数（<code style="user-select: auto;">Parameter</code>）实例作为属性，这些参数就是在训练过程中需要优化的张量（在之前的线性模型中即w和b）。模块还可以具有一个或多个子模块（<code style="user-select: auto;">nn.Module</code>的子类）属性，并且也可以追踪其参数。</p>
<blockquote style="user-select: auto;">
<p style="user-select: auto;">注：子模块必须是顶级属性（top-level attributes），而不能包含在list或dict实例中！否则，优化器将无法找到子模块（及其参数）。对于需要子模块列表或字典的情况，PyTorch提供有<code style="user-select: auto;">nn.ModuleList</code>和<code style="user-select: auto;">nn.ModuleDict</code>。</p></blockquote>

nn提供了一种通过<code>nn.Sequential</code>容器串联模块的简单方法。

得到的模型的输入是作为nn.Sequential的参数的第一个模块所指定的输入，然后将中间输出传递给后续模块，并输出最后一个模块返回的输出。

调用model.parameters()可以得到所有模块的权重和偏差。

有关nn.Modules参数的一些注意事项：当你检查由几个子模块组成的模型的参数时，可以方便地通过其名称识别参数。这个方法叫做named_parameters。实际上，Sequential中每个模块的名称都是该模块在参数中出现的顺序。有趣的是，Sequential还可以接受OrderedDict作为参数，这样就可以给Sequential的每个模块命名，此代码使你可以允许子模块有更加具有解释性的名称，你还可以通过访问子模块来访问特定的参数，就像它们是属性一样。
``` python
seq_model = nn.Sequential(
            nn.Linear(1, 13),
            nn.Tanh(),
            nn.Linear(13, 1))
print(seq_model)

print([param.shape for param in seq_model.parameters()])

for name, param in seq_model.named_parameters():
    print(name, param.shape)

from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))

print(seq_model)

for name, param in seq_model.named_parameters():
    print(name, param.shape)

print(seq_model.output_linear.bias)
```
**结果：**
``` python
Sequential(
  (0): Linear(in_features=1, out_features=13, bias=True)
  (1): Tanh()
  (2): Linear(in_features=13, out_features=1, bias=True)
)

[torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]

0.weight torch.Size([13, 1])
0.bias torch.Size([13])
2.weight torch.Size([1, 13])
2.bias torch.Size([1])

hidden_linear.weight torch.Size([8, 1])
hidden_linear.bias torch.Size([8])
output_linear.weight torch.Size([1, 8])
output_linear.bias torch.Size([1])

Parameter containing:
tensor([-0.2541], requires_grad=True)
```