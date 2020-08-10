# 从Tensor开始
## Tensor基础
张量(tensor)：将向量(vector)和数组(matrix)推广到任意维度，等价于多维数组(multidimensional array)。张量(tensor)的维度与用来索引张量中某个标量值的索引数一致。

Python列表或数字元组（tuple）是在内存中单独分配的Python对象的集合，如图左侧所示。然而，PyTorch张量或NumPy数组（通常）是连续内存块上的视图（view），这些内存块存有未封装（unboxed）的C数值类型，在本例中，如图右侧所示，就是32位的浮点数（4字节），而不是Python对象。

![](https://tangshusen.me/Deep-Learning-with-PyTorch-Chinese/img/chapter2/2.3.png "内存分配")

## Tensor与Storage
数值分配在连续的内存块中，由torch.Storage实例管理。存储（Storage）是一个一维的数值数据数组，例如一块包含了指定类型（可能是float或int32）数字的连续内存块。PyTorch的张量（Tensor）就是这种存储（Storage）的视图（view），我们可以使用偏移量和每一维的跨度索引到该存储中。

多个张量可以索引同一存储，即使它们的索引方式可能不同，如图所示。由于基础内存仅分配一次，所以无论Storage实例管理的数据大小如何，都可以快速地在该数据上创建不同的张量视图。

![](https://tangshusen.me/Deep-Learning-with-PyTorch-Chinese/img/chapter2/2.4.png "数据存储")

无法使用两个索引来索引二维张量的存储，因为存储始终是一维的，与引用它的任何张量的维数无关。

更改存储的值也会更改引用它的张量的内容。

平时使用中很少直接使用存储实例。
## 尺寸、存储偏移和步长
为了索引存储，张量依赖于几条明确定义它们的信息：尺寸（size）、存储偏移（storage offset）和步长（stride），如图所示。尺寸（或按照NumPy中的说法：形状shape）是一个元组，表示张量每个维度上有多少个元素。存储偏移是存储中与张量中的第一个元素相对应的索引。步长是在存储中为了沿每个维度获取下一个元素而需要跳过的元素数量。
![](https://tangshusen.me/Deep-Learning-with-PyTorch-Chinese/img/chapter2/2.5.png)
用下标i和j访问二维张量等价于访问存储中的storage_offset + stride[0] * i + stride[1] * j元素。偏移通常为零，但如果此张量是一个可容纳更大张量的存储的视图，则偏移可能为正值。

从最右边的维开始将其值存放在存储中的张量（例如沿着行存放在存储中的二维张量）定义为连续（Contiguous）张量。可以使用contiguous方法从非连续张量获得新的连续张量。 张量的内容保持不变，但步长发生变化，存储也是。

PyTorch中称为高级索引（advanced indexing）的功能，可以使用0/1张量来索引数据张量。此张量本质上将数据筛选为仅与索引张量中的1对应的元素（或行）。
## 数据类型
张量构造函数（即tensor、ones、zeros之类的函数）的dtype参数指定了张量中的数据类型。数据类型指定张量可以容纳的可能值（整数还是浮点数）以及每个值的字节数。
<ul style="user-select: auto;"><li style="user-select: auto;"><code style="user-select: auto;">torch.float32</code>或<code style="user-select: auto;">torch.float</code> —— 32位浮点数</li><li style="user-select: auto;"><code style="user-select: auto;">torch.float64</code>或<code style="user-select: auto;">torch.double</code> —— 64位双精度浮点数 </li><li style="user-select: auto;"><code style="user-select: auto;">torch.float16</code>或<code style="user-select: auto;">torch.half</code> —— 16位半精度浮点数</li><li style="user-select: auto;"><code style="user-select: auto;">torch.int8</code> —— 带符号8位整数</li><li style="user-select: auto;"><code style="user-select: auto;">torch.uint8</code> —— 无符号8位整数</li><li style="user-select: auto;"><code style="user-select: auto;">torch.int16</code>或<code style="user-select: auto;">torch.short</code> —— 带符号16位整数</li><li style="user-select: auto;"><code style="user-select: auto;">torch.int32</code>或<code style="user-select: auto;">torch.int</code> —— 带符号32位整数</li><li style="user-select: auto;"><code style="user-select: auto;">torch.int64</code>或<code style="user-select: auto;">torch.long</code> —— 带符号64位整数</li></ul>
<p style="user-select: auto;">每个<code style="user-select: auto;">torch.float</code>、<code style="user-select: auto;">torch.double</code>等等都有一个与之对应的具体类：<code style="user-select: auto;">torch.FloatTensor</code>、<code style="user-select: auto;">torch.DoubleTensor</code>等等。<code style="user-select: auto;">torch.int8</code>对应的类是<code style="user-select: auto;">torch.CharTensor</code>，而<code style="user-select: auto;">torch.uint8</code>对应的类是<code style="user-select: auto;">torch.ByteTensor</code>。<code style="user-select: auto;">torch.Tensor</code>是<code style="user-select: auto;">torch.FloatTensor</code>的别名，即默认数据类型为32位浮点型。</p>

## 序列化张量
PyTorch内部使用pickle来序列化张量对象和实现用于存储的专用序列化代码。

如果只想通过PyTorch加载张量，则上述例子可让你快速保存张量，但这个文件格式本身是不互通（interoperable）的，你无法使用除PyTorch外其他软件读取它。

对于需要（互通）的情况，你可以使用HDF5格式和库。HDF5是一种可移植的、广泛支持的格式，用于表示以嵌套键值字典形式组织的序列化多维数组。
## 将张量转移到GPU上
PyTorch张量的最后一点是关于在GPU上计算。每一个Torch张量都可以转移到GPU上去执行快速、大规模并且可以并行的计算。在张量上执行的所有操作均由PyTorch自带的GPU特定例程执行。

除了dtype之外，PyTorch张量还具有设备（device）的概念，这是在设置计算机上放张量（tensor）数据的位置。

可以使用速记方法<code>tensor.cpu()</code>和<code>tensor.cuda()</code>代替<code>tensor.to</code>方法来实现相同的目标。

使用<code>tensor.to</code>方法时，可以通过提供<code>device</code>和<code>dtype</code>参数来同时更改位置和数据类型。
## Tensor API
在<code>torch</code>模块下可进行张量上和张量之间的绝大多数操作，这些操作也可以作为张量对象的方法进行调用。

以上两种形式之间没有区别，可以互换使用。需要注意的是：有少量的操作仅作为张量对象的方法存在。

下划线标识表明该方法是就地（inplace）运行的，即直接修改输入而不是创建新的输出并返回；任何不带下划线的方法都将保持源张量不变并返回新的张量。