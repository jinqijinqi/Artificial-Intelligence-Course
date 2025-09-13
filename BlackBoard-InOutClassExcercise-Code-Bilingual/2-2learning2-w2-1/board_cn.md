# 板书（中文）— 机器学习 2（SGD · 步长 · 非线性特征 · 两层神经网络 · 特征模板）

## 1) 随机梯度下降（SGD）
训练损失：\( \mathrm{TrainLoss}(w)=\tfrac{1}{|D|}\sum_{(x,y)\in D}\mathrm{Loss}(x,y;w)\)。  
- **GD**：\(w \leftarrow w - \eta \nabla \mathrm{TrainLoss}(w)\)（每步遍历全数据）。  
- **SGD**：逐样本更新 \(w \leftarrow w - \eta \nabla \mathrm{Loss}(x,y;w)\)。  
- **小批量**：每步对 \(B\) 个样本的梯度求平均。

**步长**：常数（如 \( \eta=0.1\)）或递减 \( \eta_t=\eta_0/\sqrt{t}\)。大步长快但不稳，小步长稳但慢。

## 2) 非线性特征（对权重线性）
预测器 \(f_w(x)=w\cdot\phi(x)\) 对 \(w\)、\(\phi(x)\) 线性，但 \(\phi\) 可对 \(x\) 非线性：  
- 二次项：\( \phi(x)=[1,x,x^2]\)。  
- 分段/分箱：\( \phi_b(x)=[\mathbf{1}_{x\in \text{bin}_j}]_j\)。  
- 周期项：加入 \( \cos(\omega x)\) 等。

## 3) 两层神经网络
隐表示 \( h(x)=\sigma(V\,\phi(x)) \)，分数 \( s(x)=w\cdot h(x)\) → 回归/分类。  
**ReLU** \( \sigma(z)=\max(z,0)\)：正半轴梯度不消失、计算快。

## 4) 特征模板与稀疏/稠密实现
- 模板定义**一族**特征（如 “endsWith ___”、像素强度 (row,col,channel)）。  
- **稀疏字典**适合 NLP 模板；**稠密数组**适合图像像素。稀疏情形仅存非零项。
