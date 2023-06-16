Enhanced DetailNet 模型中使用的一些数学公式：

1. 卷积操作：
   - 输入特征图（H_in × W_in × C_in）与卷积核（K × K × C_in × C_out）之间的卷积运算可以表示为：
     ```math
     \text{Conv}(X, W) = \sum_{i=1}^{C_in} (X_i * W_i)
     ```
     其中，X_i 是输入特征图的第 i 个通道，W_i 是卷积核的第 i 个通道，* 表示卷积操作。
   
2. 激活函数（Activation Function）：
   - ReLU（Rectified Linear Unit）函数：
     ```
     f(x) = max(0, x)
     ```
     其中，`x` 是输入值。

3. 残差连接：
   - 输入特征图 X 通过残差连接与卷积操作的输出特征图 F 进行相加操作，表示为：
     ```math
     \text{Residual}(X, F) = X + F
     ```

4. 注意力模块：
   - 自注意力机制：自注意力机制使用了三个线性变换矩阵，分别为查询矩阵（Q）、键矩阵（K）和值矩阵（V）。注意力机制的计算公式如下：
     ```math
     \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
     ```
     其中，d_k 是查询矩阵和键矩阵的维度。

   - 多尺度注意力：多尺度注意力通过引入不同的尺度下的自注意力机制，并对结果进行融合。多尺度注意力的计算公式如下：
     ```math
     \text{MultiScaleAttention}(X) = \text{concat}\left(\text{Attention}(X; Q_1, K_1, V_1), \text{Attention}(X; Q_2, K_2, V_2), \ldots\right)
     ```
     其中，Q_i、K_i 和 V_i 分别表示第 i 个尺度下的查询矩阵、键矩阵和值矩阵。
   
5. 池化操作（Pooling）：
   - 最大池化（Max Pooling）操作：
     ```
     C[i, j] = max(A[i*s, j*s])  (i, j)
     ```
     其中，`A` 是输入特征图，`C` 是池化结果，`s` 是池化的步幅。
   
6. 全局平均池化：
   - 对最后一个卷积层的输出特征图进行全局平均池化，将特征图转换为固定长度的特征向量。全局平均池化的计算公式为：
     ```math
     \text{GlobalAvgPool}(X) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{ij}
     ```
     其中，H 和 W 分别表示特征图的高度和宽度。

这些数学公式描述了 Enhanced DetailNet 模型中的关键操作和机制，包括卷积操作、残差连接、注意力机制和全局平均池化。这些公式的应用有助于模型从输入图像中提取特征、增

强关键细节的关注、融合不同尺度的信息，并最终得到具有增强对近似物体和细节处理能力的输出结果。