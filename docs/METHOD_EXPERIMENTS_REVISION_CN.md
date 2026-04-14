# Method 与 Experiments 修订正文（中文，和当前实现对齐）

## 3. System Model and Proposed Method

### 3.1 问题定义
给定时序有符号有向图 \(G=(V,E)\)，其中每条边表示为 \((u,v,r_{uv},t_{uv})\)。若 \(r_{uv}>0\) 则标签 \(y_{uv}=1\)（trust），若 \(r_{uv}<0\) 则 \(y_{uv}=0\)（distrust），\(r_{uv}=0\) 样本剔除。目标是在时刻 \(t\) 预测边 \((u,v)\) 的信任概率 \(p(y_{uv}=1\mid u,v,t)\)，并输出模型不确定性 \(\mathcal{U}_{uv}\)。

### 3.2 严格时序协议
边按时间划分为 train/val/test，并满足
\[
\max t_{\text{train}} < \min t_{\text{val}},\quad \max t_{\text{val}} < \min t_{\text{test}}.
\]
传播图仅由 train 边构建，val/test 仅用于监督评估，避免时间泄漏。

### 3.3 有符号多跳编码器
从 train 边构建正负邻接 \(A^+,A^-\)，并采用归一化传播（实现中为带自环的对称归一化）。第 \(k\) 跳更新为
\[
H_+^{(k)}=A^+H^{(k-1)},\quad H_-^{(k)}=A^-H^{(k-1)},
\]
\[
H^{(k)}=\phi\!\left(W_f^{(k)}[W_+^{(k)}H_+^{(k)}\Vert W_-^{(k)}H_-^{(k)}]\right),
\]
保留 \(H^{(0)},\dots,H^{(K)}\) 供后续融合。

### 3.4 边级时序上下文
对候选边 \((u,v,t)\) 构建 5 维上下文向量
\[
g_{uv}(t)=\left[\Delta_u,\Delta_v,\log(1+n_{uv}^+),\log(1+n_{uv}^-),\Delta_{uv}\right],
\]
其中 \(\Delta_u,\Delta_v\) 为节点最近历史交互到当前时刻的时间差，\(n_{uv}^+,n_{uv}^-\) 为 train 前缀中的有向历史计数，\(\Delta_{uv}\) 为节点对最近一次交互时间差。无历史时采用零填充（或默认缺省值）以保证定义闭合。

### 3.5 边级证据预测与多跳融合
对每个 hop \(k\)，构建边表示
\[
z_{uv}^{(k)}=[h_u^{(k)}\Vert h_v^{(k)}\Vert |h_u^{(k)}-h_v^{(k)}|\Vert h_u^{(k)}\odot h_v^{(k)}\Vert s_{uv}^{(k)}\Vert g_{uv}(t)],
\]
其中 \(s_{uv}^{(k)}\) 包含点积与余弦相似度。证据头输出非负证据向量 \(\mathbf{e}_{uv}^{(k)}\in\mathbb{R}_{\ge 0}^2\)。采用全局 hop 权重融合：
\[
\pi_k=\frac{\exp(\beta_k)}{\sum_j \exp(\beta_j)},\quad
\mathbf{e}_{uv}=\sum_{k=0}^{K}\pi_k\mathbf{e}_{uv}^{(k)}.
\]
再构造 Dirichlet 参数
\[
\boldsymbol{\alpha}_{uv}=\mathbf{e}_{uv}+\mathbf{1},
\]
得到预测概率
\[
p_{uv}=\frac{\alpha_{uv,1}}{\alpha_{uv,0}+\alpha_{uv,1}}.
\]

### 3.6 统一不确定性定义
本文采用单一主定义
\[
\mathcal{U}_{uv}=\lambda_u\cdot \mathcal{U}_{uv}^{\text{vac}}+(1-\lambda_u)\cdot H(p_{uv}),
\]
其中
\[
\mathcal{U}_{uv}^{\text{vac}}=\frac{2}{\alpha_{uv,0}+\alpha_{uv,1}},
\]
\(H(\cdot)\) 为二元熵。该定义同时刻画“证据不足”和“决策边界模糊”。

### 3.7 训练目标（与实现一致）
采用三项损失：
\[
\mathcal{L}=\mathcal{L}_{cls}+\lambda_u\mathcal{L}_{evi}+\lambda_r\mathcal{L}_{rank}.
\]
分类项为类平衡 BCE（可选 label smoothing）：
\[
w_+=\frac{n_++n_-}{2n_+},\quad
w_-=\frac{n_++n_-}{2n_-},
\]
\[
\mathcal{L}_{cls}=-\frac{1}{|E_{tr}|}\sum_{(u,v)\in E_{tr}}w_{y_{uv}}\left[\tilde y_{uv}\log p_{uv}+(1-\tilde y_{uv})\log(1-p_{uv})\right].
\]
其中 \(\tilde y_{uv}=(1-\epsilon)y_{uv}+0.5\epsilon\)。
证据正则项 \(\mathcal{L}_{evi}\) 使用 evidential regularization；排序项为正负边对的 softplus 排序损失：
\[
\mathcal{L}_{rank}=\frac{1}{M}\sum_{i=1}^{M}\log\left(1+\exp(s_i^- - s_i^+)\right),
\]
\(s\) 为 trust logit。

---

## 4. Experiments（修订建议文本）

### 4.1 数据集与协议
实验在 `Bitcoin-OTC`, `Bitcoin-Alpha`, `Wiki-RfA` 三个时序有符号网络上进行。全部数据使用一致的前缀时序划分、train-only 传播图构建与统一评估代码。

### 4.2 对比方法
对比包含：Node2Vec+MLP、GCN、GraphSAGE、GAT、GAtrust-like、TrustGuard-like、Guardian、MC Dropout、Ours。  
其中 TrustGuard-like 使用 confidence 可直接构造 selective ranking；本文同时报告 uncertainty-based 与 confidence-based selective 结果。

### 4.3 评价指标
主分类/排序指标：AUC-PR、MCC、Macro-F1、Balanced Accuracy、ECE、Brier。  
决策导向指标：Risk-Coverage 曲线、AURC/EAURC、Selective Accuracy、不确定性过滤后风险。  
为避免争议，正文同时报告：
1. model uncertainty 排序；
2. confidence 排序（\(1-\max(p,1-p)\)）。

### 4.4 消融
消融设置：`ours`, `ours_nounc`, `ours_1hop`, `ours_notemp`。  
用于验证不确定性项、多跳传播、时间上下文模块的独立贡献，并避免把评估协议改动误当作模型增益。

### 4.5 显著性与主张边界
采用 seed-wise 配对统计（脚本中提供 exact sign-flip test）给出差异显著性。  
论文正文建议：
- 强主张仅用于“跨数据集一致、且统计显著”的结论；
- 其余只做弱主张（“在若干数据集/指标上有效”），避免“全面最优”表述。

