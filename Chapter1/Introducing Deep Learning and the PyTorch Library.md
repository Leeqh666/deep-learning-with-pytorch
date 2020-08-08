# 深度学习及PyTorch库简介
## 主要内容

* PyTroch是即时执行的，即时执行非常有用，因为如果在执行某个表达式时出现错误，Python的解释器、调试器或者其他类似的工具都能够直接访问到相关的Python对象，并且在执行出错的地方会直接抛出异常。
* PyTorch的核心数据结构-Tensor(<code>torch</code>)
* PyTorch的核心功能-自动求导(由tensor本身提供，通过<code>torch.autograd</code>完善)
* PyTorch的网络构建(<code>torch.nn</code>模块)
* Pytorch的数据加载和处理(<code>torch.util.data</code>模块)
## 补充内容
* PyTorch的分布计算-(<code>torch.nn.DataParallel</code>和<code>torch.distributted</code>)
* 使用jupyter notebook编辑交互式文档。
* PyTorch并不是唯一能处理多维数组的库。NumPy是迄今为止最受欢迎的多维数组处理库，以至于它可以被当做数据科学的通用语言。事实上，PyTorch可以与NumPy无缝衔接，从而使得PyTorch能够与Python中的其他科学库（如SciPy、Scikit-learn和Pandas）进行高度的整合。

