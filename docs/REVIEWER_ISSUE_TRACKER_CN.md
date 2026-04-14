# 审稿意见-修改对照表（代码与实验）

## 已完成

1. 仅单数据集问题  
状态：已修复。  
修改：
- 新增 `bitcoin_alpha`、`wiki_rfa` 数据加载。
- 主入口支持 `--dataset`。  
代码：
- [data/bitcoin_otc.py](/C:/Users/Administrator/Downloads/EFGNN-main/data/bitcoin_otc.py)
- [data/temporal_split.py](/C:/Users/Administrator/Downloads/EFGNN-main/data/temporal_split.py)
- [main_link.py](/C:/Users/Administrator/Downloads/EFGNN-main/main_link.py)

2. coverage 仅基于 uncertainty 的争议  
状态：已修复。  
修改：
- 增加 confidence-based selective ranking（`confidence_maxprob`, `confidence_margin`）。
- 同时导出 primary / confidence / model-uncertainty 三套 risk-coverage。  
代码：
- [eval_link.py](/C:/Users/Administrator/Downloads/EFGNN-main/eval_link.py)
- [main_link.py](/C:/Users/Administrator/Downloads/EFGNN-main/main_link.py)

3. 训练目标与论文表述一致性  
状态：已对齐。  
修改：
- 当前实现明确为 `L_cls + lambda_u * L_evi + lambda_r * L_rank`。
- 类别权重采用逆频率平衡权重（非写反版本）。  
代码：
- [train_link.py](/C:/Users/Administrator/Downloads/EFGNN-main/train_link.py)

4. 完整对比、消融、决策导向不确定性结果组织  
状态：已补齐脚本和结果导出。  
新增：
- 一键跑三数据集完整套件脚本。
- 多数据集聚合、配对显著性检验、图表提示词自动生成。  
代码：
- [scripts/run_reviewer_suite.py](/C:/Users/Administrator/Downloads/EFGNN-main/scripts/run_reviewer_suite.py)
- [scripts/compile_reviewer_results.py](/C:/Users/Administrator/Downloads/EFGNN-main/scripts/compile_reviewer_results.py)
- [scripts/export_dataset_stats.py](/C:/Users/Administrator/Downloads/EFGNN-main/scripts/export_dataset_stats.py)
- [scripts/generate_reviewer_prompts.py](/C:/Users/Administrator/Downloads/EFGNN-main/scripts/generate_reviewer_prompts.py)

5. 论文正文“方法与实验边界不清”  
状态：已提供可投稿修订文本。  
文档：
- [docs/METHOD_EXPERIMENTS_REVISION_CN.md](/C:/Users/Administrator/Downloads/EFGNN-main/docs/METHOD_EXPERIMENTS_REVISION_CN.md)
- [docs/REVIEWER_REVISION_GUIDE.md](/C:/Users/Administrator/Downloads/EFGNN-main/docs/REVIEWER_REVISION_GUIDE.md)

## 当前结果与风险（必须如实写入论文）

1. Ours 并非所有数据集、所有指标最优。  
建议：只做“在部分关键指标上显著有效”的主张，避免“全面最优”。

2. seeds=3 的统计功效有限。  
建议：若投稿前时间允许，主结果提升到 seeds=5 并补更多显著性检验。

3. Wiki-RfA 上 uncertainty-based selective 结果不稳定。  
建议：正文必须同时报告 confidence-based selective，避免单一不确定性叙事被质疑。

