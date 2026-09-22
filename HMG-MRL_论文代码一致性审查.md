# HMG-MRL 论文—代码一致性审查

审查对象：`HMG-MRL/` 当前工作区源码。

本文只依据论文描述和仓库中实际可见的源码，明确区分：

- **一致**：源码有直接证据支持。
- **代码细节**：论文正文未充分说明，但代码有明确实现。
- **差异/无法确认**：源码缺失、运行路径断裂，或计算形式与论文不同。

## 0. 结论摘要

当前仓库包含 HMG-MRL 的主要模型和预处理代码，但不能直接作为论文完整实验的可运行复现入口。

1. [run_main.py](run_main.py#L342-L351) 硬编码导入 `model.bipartite_transformer_knowledge_head_0`，该文件不存在；仓库实际存在的是 `model/bipartite_transformer.py`。
2. [other_utils.py](other_utils.py#L53-L75) 只有 `scaffold_split_valid_test` 包装函数，没有 `scaffold_split` 定义。
3. [motif_utils.py](preprocess/motif_utils.py#L313-L398) 访问 `cfg.use_brics_fragments`、`cfg.use_murcko_fragments`、`cfg.use_smarts_fragments`，但 [config.py](config.py#L1-L105) 未定义这三个属性。
4. 当前训练脚本只有分类路径：加权 `CrossEntropyLoss`、两个分类 logit、ROC-AUC；没有 regression、MSE 或 RMSE 路径。
5. test 集在每个 epoch 都被评估并写入日志，不符合 test 只在最终评估使用的严格 protocol。
6. [run_main.py](run_main.py#L273-L326) 的 `eval()` 使用了未初始化的局部变量 `batch_eval_loss`，实际评估会触发 `NameError`。
7. 模型内部确实有 atom/bond 双节点消息传递、motif 初始特征、atom-aware motif attention、descriptor 分支、三路 fusion 和 InfoNCE 风格对齐损失。
8. 没有发现五次运行的 mean/std、paired t-test、Cohen's dz、单侧置信下界，也没有六个论文 ablation 的完整执行器。

因此，论文中的结果数值不能仅凭当前仓库源码验证。

## 1. 代码结构地图

```text
入口
 ├── run_main.py
 │    ├── hyperparameter_setting
 │    ├── start_main / start_pogram
 │    ├── train_and_val
 │    ├── train / eval
 │    └── import_model
 ├── 数据处理 / 特征缓存
 │    ├── preprocess/get_atom_bond_frag_info.py
 │    ├── preprocess/Featurizer_atom_bond.py
 │    └── data/*.csv 及预期的 data/*.pickle
 ├── motif extraction
 │    └── preprocess/motif_utils.py
 ├── Bipartite Graph Encoder
 │    └── model/bipartite_transformer.py::GNN_atom_bond
 ├── Motif Transformer
 │    ├── Frag_atom_cross_attention
 │    ├── Encoder / EncoderLayer
 │    └── MultiHeadedAttention
 ├── Global Attribute Encoder
 │    ├── descriptor_mlp
 │    ├── GNN_mol_atom_aggregate
 │    └── GNN_mol_atom_update
 ├── Fusion
 │    └── Layer_attention_by_mol
 ├── Prediction
 │    └── PostGNN_classifier
 ├── L_align
 │    └── info_nce_loss
 ├── Training
 │    └── run_main.py::train_and_val / train
 └── Evaluation
      └── run_main.py::eval
```

README 对文件职责的描述见 [README.md](README.md#L25-L77)，但 README 不能替代缺失的运行代码。

## 2. 论文公式—代码位置对照表

| 论文含义 | 代码位置 | 实际实现 | 判断 |
|---|---|---|---|
| 项目入口 | [run_main.py](run_main.py#L342-L351) | 导入不存在的模型文件 | 当前不可运行 |
| atom/bond 初始 embedding | [bipartite_transformer.py](model/bipartite_transformer.py#L414-L422) | `atom_preGNN`、`bond_preGNN` 分别处理 39 和 76 维输入 | 一致于独立 embedding |
| atom/bond message passing | [bipartite_transformer.py](model/bipartite_transformer.py#L424-L454)、[bipartite_transformer.py](model/bipartite_transformer.py#L515-L534) | atom/bond 各自邻居投影、MLP、epsilon 和 residual | 核心结构对应 |
| H_b readout | [bipartite_transformer.py](model/bipartite_transformer.py#L547-L551) | 最终 atom 求和后再做 molecule-to-atom GAT attention | 只读 atom，但不只是单一 GlobalAttention |
| motif 三路提取 | [motif_utils.py](preprocess/motif_utils.py#L8-L311) | BRICS、Murcko、SMARTS functional group | 有实现，但配置开关缺失 |
| motif 合并去重 | [motif_utils.py](preprocess/motif_utils.py#L313-L398) | atom/bond 集合的 frozenset 签名全局去重 | overlap/nested 保留 |
| motif initial embedding | [get_atom_bond_frag_info.py](preprocess/get_atom_bond_frag_info.py#L154-L251) | raw atom sum + raw bond sum + concat | 基本一致 |
| Eq.(7)-(9) motif communication | [bipartite_transformer.py](model/bipartite_transformer.py#L309-L355)、[bipartite_transformer.py](model/bipartite_transformer.py#L536-L541) | concat-based attention、LeakyReLU、atom 维 softmax、motif residual | 方向一致，命名和细节不同 |
| Motif Transformer | [bipartite_transformer.py](model/bipartite_transformer.py#L45-L105) | multi-head scaled dot-product self-attention + FFN + post-norm | 代码细节，需与正文逐字核对 |
| Global Attribute Encoder | [bipartite_transformer.py](model/bipartite_transformer.py#L484-L490)、[bipartite_transformer.py](model/bipartite_transformer.py#L565-L573) | descriptor MLP 作为 query/base，与 atom 做同类 attention，再 residual | 论文没有明确写出 x_d |
| fusion | [bipartite_transformer.py](model/bipartite_transformer.py#L386-L406)、[bipartite_transformer.py](model/bipartite_transformer.py#L575-L580) | 三路求和形成 query 输入，共享 value 投影，concat score、softmax、sum residual | attention-style，但不是最简 Eq.(8) |
| prediction head | [bipartite_transformer.py](model/bipartite_transformer.py#L493-L494)、[bipartite_transformer.py](model/bipartite_transformer.py#L583-L584) | 输出 `2 * len(tasks)` 分类 logit | 仅分类 |
| alignment projection | [bipartite_transformer.py](model/bipartite_transformer.py#L473-L480)、[bipartite_transformer.py](model/bipartite_transformer.py#L547-L562) | 两个独立 Linear + ReLU projection head | 代码细节 |
| Eq.(13) 风格 loss | [bipartite_transformer.py](model/bipartite_transformer.py#L357-L382) | 2N、normalize、dot product、去 self、一个 positive、其余 negatives、CE | 数学逻辑基本一致 |
| 总损失 | [run_main.py](run_main.py#L243-L253) | classification CE + 0.15 * alignment | 仅分类 |
| optimizer/training | [run_main.py](run_main.py#L83-L109) | Adam、weight decay、scheduler | Adam 一致，额外机制未在论文摘要设置中说明 |
| split | [other_utils.py](other_utils.py#L53-L75) | 两阶段 split wrapper，但内部函数缺失 | 无法确认 |

## 3. 数据集、入口与训练流程

### 3.1 数据集覆盖

[README.md](README.md#L25-L29) 声称覆盖六个分类和三个回归数据集，但 [config.py](config.py#L7-L10) 只激活 `bace` 与单任务 `Class`。`run_main.py` 没有数据集循环、任务类型分支或 regression 分支。

脚本还依赖 `data/<task>_remained_df.pickle` 和 `data/<task>.pickle`，这些缓存文件不在当前 `data/` 文件列表中。

### 3.2 scaffold split

代码意图是先得到 train/valid/test，再把 train+valid 重新按 8:1 比例划分，见 [other_utils.py](other_utils.py#L53-L75)。但被调用的 `scaffold_split` 没有定义。因此无法确认 scaffold 分组算法、严格比例、平衡逻辑、排序规则和跨 split 是否泄漏。

五个 seed 明确列为 2020—2024，见 [run_main.py](run_main.py#L44-L55)。第一阶段使用 `random_state=cfg.seed_number`，第二阶段固定使用 8。因此按代码意图是五个不同第一阶段 split，而非固定 split 只换模型 seed；但由于函数缺失，不能执行确认。

### 3.3 随机性

[other_utils.py](other_utils.py#L12-L25) 设置 Python random、NumPy、PyTorch、CUDA 和 cuDNN deterministic。训练 batch 每个 epoch 使用 `seed + epoch` 打乱，见 [run_main.py](run_main.py#L197-L205)。代码没有 DataLoader，因此没有 worker seed 配置问题。

不过 `torch.set_default_tensor_type('torch.cuda.FloatTensor')` 把执行强绑定到 CUDA，CPU 环境不能视为支持路径。

### 3.4 test 是否只在最终评估使用

不是。`train_and_val()` 每个 epoch 都评估 train、valid、test，见 [run_main.py](run_main.py#L138-L143)，并记录 test ROC，见 [run_main.py](run_main.py#L146-L174)。保存模型的 if 条件使用 validation ROC，但 test 已被反复观察。如果据此挑选 epoch 或结果，会产生 test information leakage。

论文给出的 patience=50 也不是源码的唯一条件：代码要求 ROC 连续 50 个 epoch 且 loss 连续 20 个 epoch 都没有改进才停止，见 [config.py](config.py#L24-L26) 和 [run_main.py](run_main.py#L180-L183)。

## 4. Bipartite Graph Encoder

### 4.1 两类节点和独立 embedding

[ get_atom_bond_frag_info.py ](preprocess/get_atom_bond_frag_info.py#L269-L297) 分别创建 `atom` 和 `bond` 节点，并建立 molecule-to-atom/bond 边。atom 和 bond raw features 在 [Featurizer_atom_bond.py](preprocess/Featurizer_atom_bond.py#L76-L145) 中分别生成。

模型在 [bipartite_transformer.py](model/bipartite_transformer.py#L414-L422) 使用两个独立的 `PreGNN`。默认输入维度为 atom 39、bond 76，输出均为 256；没有调用现成 GCN/GAT 层。

### 4.2 四类 interaction

- atom 侧同时取 atom neighbor 和 bond neighbor；两者分别线性变换后相加，见 [layer_utils.py](model/layer_utils.py#L53-L80)。因此包含 atom-atom、atom-bond。
- bond 侧同时取 bond neighbor 和两端 atom neighbor；两者分别线性变换后相加，见 [layer_utils.py](model/layer_utils.py#L170-L207)。因此包含 bond-bond、bond-atom。

论文中的 W_a、W_b 在代码中不是同名单矩阵，而是 `atom_neighbor_transform[i]`、`bond_neighbor_transform[i]` 加上后续 `atom_mlps[i]`、`bond_mlps[i]`。`atom_neighbor_transform_concat` 和 `bond_neighbor_transform_concat` 被定义但 forward 没有使用。

### 4.3 epsilon、MLP 和 residual

`self.eps = nn.Parameter(torch.zeros(self.radius))`，默认 radius=2，见 [bipartite_transformer.py](model/bipartite_transformer.py#L451-L454)。更新是：

```text
atom = atom + atom_mlp((1 + eps[i]) * atom + atom_context)
bond = bond + bond_mlp((1 + eps[i]) * bond + bond_context)
```

见 [bipartite_transformer.py](model/bipartite_transformer.py#L529-L534)。因此 epsilon、MLP、residual 都存在。差异是 context 由投影后的邻居特征相加得到，并非源码中逐项显式写出论文求和式。

### 4.4 H_b readout

代码先对最终 atom embedding 按 molecule mask 求和，再调用 [GNN_mol_atom_aggregate](model/bipartite_transformer.py#L134-L183) 做 GAT-style attention，随后做 atom feature update，见 [bipartite_transformer.py](model/bipartite_transformer.py#L547-L551)。bond 最终 embedding不进入这一 readout。

因此“只由最终 atom embeddings readout”成立，但不能把它简化为论文文字中的单一 GlobalAttention 算子。

## 5. Motif extraction 与初始 embedding

### 5.1 三种 extraction

- BRICS：`BRICS.FindBRICSBonds`、`Chem.FragmentOnBonds`、`Chem.GetMolFrags`，见 [motif_utils.py](preprocess/motif_utils.py#L8-L80)。
- Bemis–Murcko：`MurckoScaffold.GetScaffoldForMol`、子结构匹配和 `Chem.FragmentOnBonds`，见 [motif_utils.py](preprocess/motif_utils.py#L82-L205)。
- Functional group：加载 [fg_dicts.txt](preprocess/fg_dicts.txt)，使用 `MolFromSmarts` 与 `GetSubstructMatches`，见 [motif_utils.py](preprocess/motif_utils.py#L207-L311)。

三种来源在 `map_molecule_to_all_fragment_types()` 合并，见 [motif_utils.py](preprocess/motif_utils.py#L313-L398)。但三个 `cfg` 开关未定义，源码当前会在调用时抛出 `AttributeError`。

### 5.2 去重、overlap、nested、多重归属

全局签名是 `(frozenset(atom_indices), frozenset(bond_indices))`。因此：

- exact duplicate 删除；
- 非完全相同的 overlap/nested motif 保留；
- 没有强制 motif 互斥；一个 atom/bond 可以属于多个 motif；
- `whole_mol` 始终存在，所以三种方法都无结果时仍至少有一个 token。

这与用户给出的 motif 规则一致。whole molecule 是否应作为 Transformer 的 motif token，需要从论文正文或补充材料另外确认。

### 5.3 motif initial embedding

[calculate_feature_vector](preprocess/get_atom_bond_frag_info.py#L212-L251) 对 motif 内 raw atom features 求和、raw bond features 求和后 concat。模型再用 `frag_input_projection` 将 115 维投影到 256 并 ReLU，见 [bipartite_transformer.py](model/bipartite_transformer.py#L465-L466) 和 [bipartite_transformer.py](model/bipartite_transformer.py#L536-L541)。

必须区分：

- x_m：raw atom/bond initial features 的 sum + concat；
- `frag_features`：x_m 投影并激活后的输入；
- `atom_feature`：atom 经 bipartite message passing 后的 learned embedding。

代码没有用 message-passing 后 atom embedding 构造 motif initial embedding，而是在 CGComm 中把最终 atom embedding作为 value；这与论文对 initial feature 和最终 atom embedding 的区分相符。

## 6. CGComm / Eq.(7)-(9)

### 6.1 逐行对应

论文形式为：

```text
Q_k^m = W_q x_k^m
V_i^m = W_v h_i^(a,L)
score = LeakyReLU(z^T [Q || V])
gamma = softmax(score over atoms)
h_k^m = x_k^m + sum(gamma * V)
```

代码在 [Frag_atom_cross_attention.forward](model/bipartite_transformer.py#L309-L355) 中实现：

- `mol_atom_feature` 是 motif feature，经过 `mol_atom_fc`，可对应 Q。
- `atom_feature` 是最终 atom embedding，经过 `mol_atom_neighbor_fc`，可对应 V。
- `torch.cat` 明确使用 concat compatibility。
- `mol_GAT_align` 将 concat 结果变成标量，相当于可学习 score vector/linear layer。
- 默认 `GNN_attn_act=lrelu`，对应 LeakyReLU 风格激活。
- `F.softmax(..., dim=2)` 沿 atom 维归一化。
- weighted atom context 求和后，forward 执行 `frag_features = frag_features + frag_context`，即 residual。

论文没有显式定义 K，代码也没有单独 key；它直接以 Q 和 V 的 concat 打分。因此不是标准 Q-K dot-product attention，而是 GAT-style concat attention。

### 6.2 A=20、K=5、D=128 的形状

| 张量 | 形状 |
|---|---|
| motif input x_m | `(K,D)` = `(5,128)` |
| final atom input h | `(A,D)` = `(20,128)` |
| Q projection | `(5,128)` |
| V projection | `(20,128)` |
| broadcast concat | `(5,20,256)` |
| score | `(5,20,1)` 或 `(5,20)` |
| gamma | `(5,20)`，沿 A softmax |
| weighted atom context | `(5,128)` |
| residual motif output | `(5,128)` |

实际 batch 前面还有 batch 维，atom/motif 数量通过 padding mask 对齐。

## 7. Global Attribute Encoder：论文中 x_d 的疑问

### 7.1 descriptor 来源与标准化

预处理使用：

```python
descriptor = dc.feat.RDKitDescriptors(is_normalized=True)
```

见 [get_atom_bond_frag_info.py](preprocess/get_atom_bond_frag_info.py#L423-L431) 和 [get_atom_bond_frag_info.py](preprocess/get_atom_bond_frag_info.py#L610-L617)。每个 SMILES 调用 `descriptor.featurize(smiles)[0]`，见 [get_atom_bond_frag_info.py](preprocess/get_atom_bond_frag_info.py#L531-L531) 和 [get_atom_bond_frag_info.py](preprocess/get_atom_bond_frag_info.py#L717-L717)。模型将输入维度硬编码为 200，见 [bipartite_transformer.py](model/bipartite_transformer.py#L484-L490)。

代码使用 DeepChem 2.8.0 的 RDKitDescriptors 集合和 `is_normalized=True`，而不是源码自己列出 MolWt/TPSA/MolLogP 等 descriptor 名称。当前执行环境没有安装 deepchem，仓库也没有保存 descriptor name list，因此完整名称需要在锁定的 DeepChem 2.8.0 环境中另行确认。

更重要的是，代码没有按 training set 计算 mean/std，也没有复用 train scaler 到 validation/test；它使用 featurizer 的内置 normalized 行为。论文若明确要求 train mean/std，则这是一个实际差异：

> 论文正文写的是 train mean/std 标准化；代码采用 `RDKitDescriptors(is_normalized=True)`，两者不能直接视为一致。

### 7.2 descriptor 到 atom 的通信

1. descriptor 经过 `descriptor_mlp`：默认 200→256→256，包含激活和 dropout，见 [bipartite_transformer.py](model/bipartite_transformer.py#L484-L490)。
2. descriptor 输出传给 `GNN_mol_atom_context_des`，与 final atom embedding 做同类 concat attention，见 [bipartite_transformer.py](model/bipartite_transformer.py#L567-L573)。
3. context 与 descriptor query 相加，并按 `update_type='skipsum'` 激活，形成 `mol_feature_by_des`，见 [bipartite_transformer.py](model/bipartite_transformer.py#L571-L573)。

所以代码中的 x_d 实际是 descriptor MLP 输出的 query/base representation。准确结论是：

> 论文正文没有说明这一点，代码采用 descriptor 经 MLP 后作为 base/query，并执行 `descriptor_query + weighted_atom_context` 的 residual update。

最终 H_d 不是单独 weighted atom values，也不是 CLS，而是 residual update 后的 `mol_feature_by_des`。

## 8. Fusion：attention 还是 gating？

三路表示是 `[frag_cls, mol_feature_by_atom, mol_feature_by_des]`，见 [bipartite_transformer.py](model/bipartite_transformer.py#L575-L576)。代码随后：

1. 在粒度维求和得到 `mol_layer_feature_by_atom_sum`，作为 fusion query 的输入。
2. [Layer_attention_by_mol](model/bipartite_transformer.py#L386-L406) 对 query 和三路 value 使用线性投影。
3. concat 后经 LeakyReLU 和标量 Linear 得到 score。
4. `F.softmax(..., -2)` 沿三个粒度归一化。
5. softmax 权重对三路 value 加权求和。
6. 返回前又执行 `mol_layer_context + mol_layer_feature_by_atom_sum` 并激活，见 [bipartite_transformer.py](model/bipartite_transformer.py#L577-L580)。

因此，从实际计算看，它是 attention-style weighted aggregation，同时带有 sum residual；不是 sigmoid gate、不是普通 scalar gate。称为 attention 有代码依据。

但它不严格等于论文 Eq.(8) 的最简写法：query 来自三路求和，value 使用共享 projection，score 由 Linear 实现，且有额外 residual/activation。论文正文没有说明这些额外操作。

最终判断：**更接近 attention，而不是 gating；但论文描述不足以唯一推出当前实现。**

## 9. L_align / Eq.(13)

### 9.1 2N、positive、negative

batch size 为 N 时：

```text
mol_feature_by_atom_projH: (N,D)
frag_cls_projH:            (N,D)
features = cat(..., dim=0): (2N,D)
```

两种 view 是 molecule atom/global branch 和 motif CLS branch，见 [bipartite_transformer.py](model/bipartite_transformer.py#L547-L562)。正样本是另一 view 中同 batch index 的 feature。

`info_nce_loss()` 先 L2 normalize，计算完整 pairwise dot product，去掉 diagonal，将同 index 配对作为 positive，其余非 self feature 作为 negatives，见 [bipartite_transformer.py](model/bipartite_transformer.py#L357-L382)。

### 9.2 是否双向、是否像交叉熵

是双向的：2N 行全部作为 anchor。前 N 行的 positive 是后 N 行同 index；后 N 行的 positive 是前 N 行同 index；每个 anchor 有 `2N-2` 个 negatives。

代码把 positive 放在 logits 第 0 位，把其余相似度放在后面，然后使用 `CrossEntropyLoss(logits / temperature, label=0)`。因此它正是“在 batch 其它 feature 中分类出同分子 positive”的 softmax + cross-entropy 形式。

### 9.3 代码参数与论文差异

- projection head：两个独立 `Linear(256,256,bias=False) + ReLU`。
- normalization：在 `info_nce_loss()` 内完成。
- self pair：排除。
- temperature tau：0.15，见 [config.py](config.py#L95-L96)。
- alignment coefficient alpha：0.15，见 [config.py](config.py#L95-L96)。
- 训练总损失：分类 CE + alpha * alignment，见 [run_main.py](run_main.py#L243-L253)。
- 代码没有 regression 对应的 prediction loss。

## 10. Loss、优化器与训练设置

| 项目 | 代码事实 | 判断 |
|---|---|---|
| optimizer | Adam | 与论文一致 |
| learning rate | hyperparameter 中为 1e-3 | 初始设定一致，但 scheduler 会改变轨迹 |
| batch size | 64 | 与论文一致 |
| early stopping | ROC 50 且 loss 20 的 AND 条件 | 不等于单一 patience 50 |
| classification criterion | 加权 CrossEntropyLoss，2 logits | 不是 BCE/BCEWithLogitsLoss |
| regression criterion | 未实现 | 明确缺失 |
| dropout | 0.3 | 代码细节 |
| weight decay | 1e-5 | 论文正文未说明 |
| scheduler | 默认 Noam，也有 ReduceLROnPlateau 分支 | 论文正文未说明 |
| alpha | 0.15 | 代码细节 |
| tau | 0.15 | 代码细节 |

`run_main.py` 虽然导入了 `mean_squared_error` 等回归指标，但没有调用；导入不等于实现。

## 11. 评价指标与统计结果

### 11.1 分类 ROC-AUC

代码只保留 label 为 0 或 1 的样本，再逐 task 调用 `roc_auc_score`。这对 missing label 的过滤思路是合理的，但没有处理某 split/task 只剩一个类别时的异常。

`eval_roc` 是逐 task AUC 列表，训练循环对该列表取 mean，见 [run_main.py](run_main.py#L144-L146)。意图上接近 task-wise macro average，而不是把所有样本拼成 micro AUC；但当前 config 只有一个 task，无法验证 ClinTox、SIDER、Tox21 的实际宏平均。

### 11.2 RMSE

当前没有 regression 分支，也没有 sqrt(MSE) 执行路径。ESOL、FreeSolv、Lipophilicity 的 RMSE 无法由当前训练代码复现。

### 11.3 五次运行与统计检验

五个 seed 会启动五组超参数组合，见 [run_main.py](run_main.py#L44-L81)，但源码没有：

- 收集五次 test 指标的结构化结果；
- 跨 run 计算 mean ± std；
- one-sided paired t-test；
- P values；
- Cohen's dz；
- one-sided 95% lower confidence bounds。

代码中的 mean 只是当前 epoch 的多任务 ROC 或 batch loss 平均，不是论文结果表的五次独立运行统计。

## 12. Ablation Study

论文的六个 variant 是：

1. w/o Bipartite Graph Encoder
2. w/o Motif Transformer
3. w/o Molecular Descriptor
4. w/o CGComm Module
5. w/o Alignment Loss
6. w/o CGComm Module and Alignment Loss

当前没有发现对应的 ablation runner、variant registry 或结果汇总代码。

motif_utils 中的三个来源开关理论上可以关闭 BRICS/Murcko/SMARTS，但不是六个论文 variant 的实现，而且配置属性缺失。将 `cfg.ct_loss_coff=0` 可以局部模拟 w/o Alignment Loss，但源码没有独立入口，也没有证明其余结构完全不变。

无法确认：

- w/o CGComm 是否只删除 cross-attention；
- w/o Molecular Descriptor 是否完全删除 descriptor branch；
- w/o CGComm + Alignment 是否仍保留三个 granularity encoder；
- w/o Bipartite/Motif Transformer 是否有等价替代 encoder。

结论：**未发现六个 ablation 可直接运行的完整实现。**

## 13. 论文—代码差异清单

### A. 与论文明确一致或高度对应

- atom 和 bond 是独立节点类型，并分别 embedding。
- message passing 同时包含 atom-atom、atom-bond、bond-bond、bond-atom。
- 有可学习 epsilon、MLP 和 residual update。
- 三种 motif 来源：BRICS、Murcko、SMARTS functional group。
- motif 使用 raw atom/bond features 的 sum + concat。
- motif 与最终 atom embedding 做 concat-based attention。
- attention 沿 atom 维 softmax，motif 有 residual。
- alignment 形成 2N features，同分子跨 view positive，L2 normalization 和温度缩放。
- optimizer 是 Adam，batch size 是 64。

### B. 论文正文未明确、代码有具体实现

- atom/bond input dim 为 39/76，hidden dim 为 256。
- bipartite radius 为 2。
- motif encoder 为 2 层、8-head、dropout 0.3。
- whole molecule 是第 0 个 motif token，使用 `frag_output[:,0,:]`。
- descriptor 输入硬编码 200，来自 DeepChem `RDKitDescriptors(is_normalized=True)`。
- descriptor MLP 为 200→256→256，并采用 descriptor query + atom context residual。
- projection head 为独立 Linear 256→256 + ReLU。
- fusion query 由三路求和得到；value 使用共享 projection；输出有 sum residual 和 activation。
- alpha=0.15、tau=0.15。
- weight decay=1e-5，默认有 scheduler。
- 分类使用 weighted CrossEntropyLoss。

### C. 论文描述与代码可能不一致

| 论文写什么 | 代码做什么 | 影响 | 判断 |
|---|---|---|---|
| train mean/std 标准化 descriptor | DeepChem `is_normalized=True`，没有 train-only scaler | 改变 descriptor 分布和性能 | 明确差异 |
| Global Attribute 沿用 Eq.(7)-(9) | descriptor MLP 输出作为 query/base，再 query + atom context residual | 论文未定义 x_d | 论文不完整 |
| H_b=GlobalAttention(atom) | atom sum 后再 GAT-style attention 和 update | readout 不同 | 可能差异 |
| fusion 为 Eq.(8) attention | 有 concat score、LeakyReLU、三项 softmax，但还有 sum residual/activation | 影响 fusion 数值 | 部分一致 |
| patience=50 | ROC 50 AND loss 20 | 停止时机不同 | 明确差异 |
| test 最终评估 | 每 epoch 评估并记录 test | test 观察泄漏风险 | 明确差异 |
| 九个数据集、分类+回归 | 当前仅 bace 分类 | 无法复现完整实验 | 明确缺失 |
| 五次报告 mean±std | 只有 seed 循环，无汇总 | 无法得到论文表 | 明确缺失 |
| 六个 ablation | 无完整 runner | 无法验证消融 | 明确缺失 |
| 自定义 scaffold split | wrapper 调用未定义函数 | 训练不可运行 | 致命缺失 |
| 正式模型入口 | 硬编码模块不存在 | 训练不可运行 | 致命缺失 |

## 14. 关键运行阻断点

按执行顺序：

1. 模型入口文件不存在。
2. `scaffold_split` 缺失。
3. motif 配置开关缺失。
4. 当前环境没有 deepchem，且代码默认 CUDA tensor。
5. `eval()` 的 `batch_eval_loss` 未初始化。
6. 即使修复上述问题，仍只有分类路径，不能生成九个数据集结果。

静态编译通过不能发现缺少模块、缺少配置属性或未初始化局部变量等运行时问题。

## 15. 复现论文必须确认的参数清单

| 参数 | 当前代码 | 是否需外部确认 |
|---|---|---|
| 数据集和 task schema | config 只有 bace/Class | 是 |
| scaffold split 算法 | `scaffold_split` 缺失 | 必须补齐/确认 |
| 8:1:1 是否严格按 scaffold group | wrapper 仅表达意图 | 是 |
| 五次 seed | 2020—2024 | 需确认 |
| 是否固定 split | 第一阶段随 seed 变化 | 需确认 |
| atom/bond feature dim | 39/76 | 代码明确 |
| hidden dimension | 256 | 代码明确 |
| radius/layers | radius=2，motif encoder=2 | 代码明确 |
| motif extraction | 三种实现，开关缺失 | 必须确认 |
| descriptor list | DeepChem RDKitDescriptors，200 维输入 | 必须从锁定环境确认名称 |
| descriptor standardization | 不是 train mean/std | 必须确认论文和官方代码优先级 |
| alpha | 0.15 | 代码明确，正文需核对 |
| tau | 0.15 | 代码明确，正文需核对 |
| projection head | Linear 256→256 + ReLU | 代码明确 |
| fusion | concat + LeakyReLU + Linear + softmax + residual | 代码明确，正文不完整 |
| optimizer/lr | Adam，hyperparameter lr=1e-3 | scheduler 后轨迹需确认 |
| batch size | 64 | 代码明确 |
| early stopping | ROC 50 AND loss 20 | 与 patience=50 不完全一致 |
| regression/RMSE | 未实现 | 必须补充/确认 |
| mean/std | 未实现 | 必须补充 |
| paired t-test、dz、CI | 未实现 | 必须补充 |
| 六个 ablation | 未实现完整 runner | 必须补充 |

## 16. 最终判断

### A. 可以确认的核心思想

源码确实体现了层次化多粒度结构：独立 atom/bond 表示、显式 bipartite message passing、motif 多源提取、raw motif initial embedding、motif-to-atom communication、descriptor-to-atom communication、三路表示融合和双视图 InfoNCE 风格对齐。

### B. 不能据此声称严格复现的部分

九个数据集、scaffold split、train-only descriptor 标准化、回归任务、最终 test protocol、五次 mean±std、统计检验和六个 ablation 都没有被当前源码完整证明。入口、split 和评估函数还存在运行阻断点。

### C. 三个重点问题的直接回答

- **Global Attribute 的 x_d**：论文正文没有明确给出；代码用 descriptor MLP 输出作为 base/query，加 atom weighted context 后 update。
- **Fusion 的 beta_n**：代码对三个粒度的 concat-based score 做 softmax，不是普通 sigmoid gate；但增加了 sum residual 和 activation，不能说与 Eq.(8) 完全逐字等价。
- **L_align 为什么像交叉熵**：代码把每个 anchor 的 positive 放在 logits 第 0 位，把其它非 self feature 作为 negatives，再用 CrossEntropyLoss；2N 行全部作为 anchor，因此是双向计算。

在修复入口、split、配置开关和评估运行错误，并锁定 DeepChem/RDKit 版本及 descriptor 名称之前，不建议把论文表中的结果归因于当前仓库的可复现实验。
