# Experiments 正文草稿（Bitcoin-OTC / Bitcoin-Alpha）

## 4. Experimental Setup

### Datasets
我们在两个经典时序有符号信任网络上评估方法：
`Bitcoin-OTC` 与 `Bitcoin-Alpha`。  
两者都采用统一的严格时间划分：`train < val < test`，并且传播图仅由训练前缀边构建，以避免时间泄漏。

### Baselines
对比方法包括：
Node2Vec+MLP、GCN、GraphSAGE、GAT、GAtrust-like、TrustGuard-like、Guardian、MC Dropout 与 Ours。

### Metrics
我们报告：
`AUC-PR`, `MCC`, `ECE`, `Brier`，以及决策导向指标 `AURC(confidence)`。  
其中 `AUC-PR` 与 `MCC` 用于评估不平衡信任边分类质量，`AURC(confidence)` 评估 selective prediction 的风险-覆盖质量（越低越好）。

---

## 5. Main Results

### Bitcoin-OTC
Ours 达到 `AUC-PR=0.9145`, `MCC=0.1596`, `AURC(conf)=0.1532`, `ECE=0.0496`。  
相较 TrustGuard-like，Ours 在 AUC-PR 与 MCC 分别提升 `+0.0227` 与 `+0.0933`，同时 AURC(conf) 降低 `0.0495`。  
相较 GAtrust-like，Ours 在 AUC-PR 与 MCC 分别提升 `+0.0182` 与 `+0.0936`，AURC(conf) 降低 `0.0504`。  
该数据集上，Ours 在 AUC-PR 与 ECE 上处于最优水平，显示出较强的排序能力和较好的校准质量。

### Bitcoin-Alpha
Ours 达到 `AUC-PR=0.8706`, `MCC=0.0709`, `AURC(conf)=0.2135`。  
相较 TrustGuard-like，AUC-PR 与 MCC 分别提升 `+0.0401` 与 `+0.1492`，AURC(conf) 降低 `0.2334`。  
相较 GAtrust-like，AUC-PR 与 MCC 分别提升 `+0.0384` 与 `+0.1224`，AURC(conf) 降低 `0.1411`。  
在与通用 GNN 基线比较时，Ours 在 AUC-PR 上接近最优（接近 GCN），并在风险过滤指标上保持竞争力。

### Claim Boundary
本文不声称“所有指标全面最优”，而强调：
1. Ours 在两个 Bitcoin 数据集上对 trust-specific 强基线（TrustGuard-like, GAtrust-like）在 `AUC-PR/MCC` 上均有稳定优势。  
2. Ours 在关键决策指标 `AURC(conf)` 上显著优于 trust-specific 基线，体现了不确定性/置信度用于决策过滤的有效性。  
3. 对通用 GNN 基线，Ours 为“top-tier”而非“uniformly best”。

---

## 6. Ablation Study

我们在两数据集上比较 `ours`, `ours_nounc`, `ours_1hop`, `ours_notemp`。

### Bitcoin-OTC
`ours_1hop` 的 AUC-PR 与 AURC 明显退化（AURC 从 `0.1532` 升至 `0.2862`），说明多跳传播对性能稳定性至关重要。  
`ours_nounc` 在 AUC-PR 略高于 ours，但 MCC 与 AURC(conf) 劣于完整模型，说明不确定性建模在决策质量和稳健性上提供了价值。  
`ours_notemp` 在 ECE 略优，但在 AUC-PR/MCC/AURC 上不及完整模型，表明时间上下文仍对总体效果有贡献。

### Bitcoin-Alpha
完整模型在 AUC-PR 与 MCC 上均明显优于三个消融版本：  
对 `ours_nounc`：AUC-PR 提升约 `+0.0308`，MCC 提升约 `+0.0848`；  
对 `ours_1hop`：AUC-PR 提升约 `+0.0204`，MCC 提升约 `+0.0904`；  
对 `ours_notemp`：AUC-PR 提升约 `+0.0135`，MCC 提升约 `+0.0346`。  
该结果支持完整模型中不确定性项、多跳传播和时间上下文模块的联合有效性。

---

## 7. Compared with Previous Version

与上一版相比，本版的主要改进如下：
1. 仅保留 Bitcoin-OTC 与 Bitcoin-Alpha 两个核心数据集，实验叙事更聚焦。  
2. 通过自动调参与重跑，Ours 在两个数据集上的 `AUC-PR/MCC` 均提升；其中 OTC 提升最明显。  
3. 决策导向指标统一采用 `AURC(confidence)` 作为主指标，避免“只用 uncertainty 排序”引发的协议争议。  
4. 消融结果更清晰，尤其在 Alpha 上能够完整支撑模块贡献结论。

