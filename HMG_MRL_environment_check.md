# HMG-MRL Environment Verification

审查范围：当前工作区中全部 Python 源码和 `environment.yml`。本报告只做核验，不修改源码或环境文件。结论区分“代码直接 import/调用”和“环境文件中存在但当前代码没有使用”；没有因为包名称看起来相关就把它列为必需依赖。

## 1. Overall Conclusion

**结论：不能直接使用，需要修改环境文件后再创建。**

原因不是项目模型代码要求 torchvision、DGL 或 UMAP，而是当前 `environment.yml` 不能直接作为 Windows 上的可移植环境文件：

1. 文件包含 Linux 构建包及 Linux 专属版本构建信息，例如 `ld_impl_linux-64`、`libgcc-ng`、`libgomp`、`libstdcxx-ng`，并固定了 `prefix: /root/miniconda3`。
2. 运行入口 `run_main.py` 直接 `import hyperopt`，但 `environment.yml` 的 pip 列表没有 `hyperopt`。当前终端虽然已经执行过 `pip install hyperopt`，这不改变环境文件本身缺失依赖的事实。
3. `torch==2.0.0+cu118` 与 `torchvision==0.15.1+cu118` 的版本配对本身合理，但文件没有声明 PyTorch CUDA wheel 所需的额外 PyTorch pip index；在干净环境中不能仅根据该文件保证这些带 `+cu118` 的 wheel 能被 pip 找到。
4. `torchdata==0.8.0` 与 PyTorch 2.0.0 不是该项目代码需要的组合，且按 torchdata 的版本分代关系存在明显的版本代际错配风险。项目没有 import `torchdata`，因此它不是当前代码运行的必要依赖。
5. 独立于环境文件，当前仓库的训练入口还存在代码级阻断：入口硬编码导入不存在的 `model.bipartite_transformer_knowledge_head_0`；`scaffold_split` 未定义；motif 配置开关未在 `config.py` 定义；这些不是通过安装依赖可以修复的问题。

因此，若“直接使用”是指从该文件在 Windows 干净环境中创建并运行当前仓库，答案为否。若在作者原 Linux 机器、已有对应本地缓存和额外手工安装的前提下，文件可能接近其历史实验环境，但仍不能证明它覆盖当前仓库入口的全部运行要求。

## 2. Original Environment Summary

| 项目 | `environment.yml` 中的值 | 核验 |
|---|---|---|
| 环境名 | `base` | 可创建但不适合作为项目专用环境；会污染或覆盖 Conda 基础环境 |
| Python | `3.8.10` | 与当前源码语法和直接使用的 API 没有明显冲突；属于较旧但可复现实验的版本 |
| PyTorch | `2.0.0+cu118` | CUDA 11.8 pip 构建；代码大量使用 CUDA，实际运行需要可用 NVIDIA 驱动/GPU |
| torchvision | `0.15.1+cu118` | 与 torch 2.0.0 的同代配对合理；当前源码不 import |
| torchdata | `0.8.0` | 当前源码不 import；与 torch 2.0.0 存在明显代际错配风险，不应作为本项目依据 |
| DGL | `2.1.0` | 当前源码不 import；不能从代码证明需要 |
| DeepChem | `2.8.0` | `get_atom_bond_frag_info.py` 使用 `dc.feat.RDKitDescriptors(is_normalized=True)`，属于直接必需依赖 |
| RDKit | `2022.9.5` | 大量用于 SMILES、3D 构象、原子/键特征、BRICS、Murcko、SMARTS；直接必需 |
| NumPy | `1.24.2` | 大量数组、随机数、特征和张量准备；直接必需 |
| Pandas | `1.3.5` | 数据表切分、拼接、CSV 输出；直接必需 |
| SciPy | `1.10.1` | Python 源码没有直接 import；可能是科学计算库的间接依赖，不是代码直接必需 |
| scikit-learn | `1.2.2` | scaffold 辅助流程和多种分类/回归指标 import；直接必需 |
| Matplotlib | `3.7.1` | 预处理模块直接 import 并设置 `agg` backend，虽然当前主流程中绘图调用不明显；模块导入仍要求安装 |
| tqdm | `4.61.2` | `other_utils.py` 直接 import；当前已导入但可见路径中使用很少 |
| hyperopt | 未列出 | `run_main.py` 第 1、2、4 行直接 import；这是明确缺失的直接依赖 |
| CUDA/平台基础包 | `*_linux-64`、`libgcc*` 等 | 作者 Linux 环境快照，不能原样用于 Windows |

代码还固定使用 `torch.cuda.FloatTensor`、`torch.cuda.LongTensor`，并在入口调用 `torch.set_default_tensor_type('torch.cuda.FloatTensor')`。所以这不是一个经过代码验证的 CPU 环境；即使 Python 依赖安装成功，无 CUDA GPU 的运行仍会失败。

## 3. Dependencies Actually Used by the Code

下表以所有 `*.py` 文件的 import 和实际 API 调用为依据。文件位置采用仓库相对路径。

| 包/模块 | 是否直接 import | 使用位置和用途 | environment.yml | 版本判断 |
|---|---|---|---|---|
| `torch` / `torch.nn` | 是 | `config.py`、`model/*.py`、`other_utils.py`、`run_main.py`；模型层、CUDA 张量、优化器、损失、保存模型 | 是，`torch==2.0.0+cu118` | 使用的 `nn.Linear`、`LayerNorm`、`Multihead` 之外的基础张量/API 与 2.0 兼容；CUDA 路径要求 GPU |
| `numpy` | 是 | 所有特征数组、随机打乱、RBF、指标输入 | 是，`1.24.2` | 与源码使用方式匹配；未发现要求更高版本的 API |
| `pandas` | 是 | `other_utils.py` 中 `concat`、筛选、采样；`run_main.py` 中 CSV 输出 | 是，`1.3.5` | 与调用方式匹配 |
| `scikit-learn` | 是 | `other_utils.py` 的 metrics、`train_test_split`；`run_main.py` 的 KFold 和指标 | 是，`1.2.2` | 与直接调用匹配；`scaffold_split` 本身并未由 sklearn 提供 |
| `rdkit` | 是 | `preprocess/Featurizer_atom_bond.py`、`motif_utils.py`、`get_atom_bond_frag_info.py`、`other_utils.py`；分子解析、特征、3D 构象、片段和描述相关处理 | 是，`2022.9.5` | 代码使用的 `Chem`、`AllChem`、`BRICS`、`Recap`、`MurckoScaffold` 等 API 属于该代 RDKit 的常规 API |
| `deepchem` | 是 | `preprocess/get_atom_bond_frag_info.py`；`dc.feat.RDKitDescriptors(is_normalized=True)` 生成约 200 维描述符 | 是，`2.8.0` | 与代码调用形式匹配；DeepChem 与 RDKit/NumPy 的组合应保持原版本，不建议无理由升级 |
| `matplotlib` | 是 | `get_atom_bond_frag_info.py` 导入 pyplot、cm、backend 和 RDKit drawing 相关支持 | 是，`3.7.1` | 满足当前 import；模块顶层会执行 `plt.switch_backend('agg')` |
| `tqdm` | 是 | `other_utils.py` 导入 `tqdm` | 是，`4.61.2` | 满足当前 import |
| `hyperopt` | 是 | `run_main.py` 导入 `Trials`、`hp`、`fmin`、`tpe`、`STATUS_OK`，当前代码实际主要定义/保留超参数搜索接口 | **否** | 必须显式补入；当前已由终端手工安装，但未记录在 yml |
| `torchvision` | 否 | 全部 Python 文件没有 import 或 API 调用 | 是，`0.15.1+cu118` | 版本配对合理，但对当前仓库不是必需依赖 |
| `dgl` | 否 | 全部 Python 文件没有 import；模型消息传递由项目自定义 PyTorch 模块实现 | 是，`2.1.0` | 不能由当前代码证明需要 |
| `torchdata` | 否 | 全部 Python 文件没有 import | 是，`0.8.0` | 对当前代码非必要；与 torch 2.0 的代际组合不建议保留为核心依据 |
| `umap` / `umap-learn` | 否 | 全部 Python 文件没有 import | 是，两个包均有 | 当前项目不使用；`umap` 还是另一个同名包，容易造成包名混淆 |
| `wandb` | 否 | 没有 import 或 API 调用 | 是，`0.19.11` | 当前项目不使用 |
| `tensorboard` | 否 | 没有 import 或 SummaryWriter/API 调用 | 是，`2.12.0` | 当前项目不使用 |

未发现源码直接 import SciPy、seaborn、joblib、networkx、pubchempy、PIL、requests 或 yaml。它们只能按间接依赖、预留工具或导出环境中的其他用途处理，不能列入代码必需直接依赖。

## 4. Dependency Coverage

### A. 必需依赖

当前源码直接需要：`python`、`torch`、`numpy`、`pandas`、`scikit-learn`、`rdkit`、`deepchem`、`matplotlib`、`tqdm` 和 `hyperopt`。其中前九项在 yml 中有声明，`hyperopt` 缺失。严格按当前入口的 import 链，缺少 `hyperopt` 就足以使 `run_main.py` 在启动时失败。

此外，代码运行还依赖仓库/数据文件而非 pip 包：`preprocess/Electronegativity.pkl`、`preprocess/fg_dicts.txt`，以及 README 所述但当前工作区没有找到的 `data/<task>.pickle` 和 `data/<task>_remained_df.pickle`。这些不是环境依赖，但会影响“环境创建成功后能否运行”。

### B. 间接依赖

`scipy`、`joblib`、`threadpoolctl`、`networkx`、`pillow`、`python-dateutil`、`pytz`、`packaging`、`requests` 等可能由 NumPy/Pandas/scikit-learn/Matplotlib/DeepChem/Jupyter 等包带入或使用，但当前 Python 代码没有直接 import。它们不需要因为本项目源码而单独增加；保留在导出文件中也不等于本项目直接需要。

### C. 非必要或当前代码未使用

以下包在文件中存在，但对当前 Python 源码没有直接使用证据：

- `torchvision`、`torchdata`、`dgl`；
- `umap==0.1.1`、`umap-learn==0.5.7`；
- `wandb`、`tensorboard` 及 TensorBoard 周边；
- Jupyter 全套：`jupyterlab`、`notebook`、`ipykernel`、`ipywidgets`、`nbconvert` 等；
- 绘图周边 `seaborn`；
- 实验/服务周边 `docker-pycreds`、`gitpython`、`supervisor`、`sentry-sdk`、`setproctitle` 等；
- 与当前源码没有直接 import 证据的 `pubchempy`、`py4j`、`google-auth*`、`aiofiles`、`aiosqlite` 等。

这些只被列为可能不必要，不建议在本次核验中擅自删除，因为它们可能来自作者未提交的 notebook、工具脚本或历史实验流程。

## 5. Version Compatibility

### Python 3.8.10

当前源码没有使用超过 Python 3.8 的语法；直接依赖也没有从代码表现出必须更高 Python 版本的 API。作为论文复现版本可以保留。需要注意的是，文件中很多 pip 包是 2023--2025 年版本，未来在 Python 3.8 上重新解析时，pip/Conda 是否仍提供对应 wheel 不能由源码保证。

### PyTorch 2.0.0+cu118 and torchvision 0.15.1+cu118

这两个版本是同一代的官方配对，未发现版本号本身冲突。源码使用 `torch.nn` 基础层、张量操作、Adam、学习率调度器和 CUDA tensor 类型，未调用明显晚于 2.0 的 API。

实际风险在安装来源和运行平台：`+cu118` 是 pip local-version 标记，环境文件没有 `--extra-index-url https://download.pytorch.org/whl/cu118` 或等价配置。Conda 的 `pytorch` channel 也不会自动把这两个 pip 条目转换成 Conda 包。因此干净环境中必须验证 pip 是否能取得这两个准确 wheel；在 Windows 上还必须有匹配的 NVIDIA 驱动。不要仅因版本号看起来合理就认为 CUDA 已安装完成。

### torchdata 0.8.0

项目没有 import `torchdata`，所以它不参与当前模型执行。若把它视为框架核心，则 `0.8.0` 属于较新的 torchdata 代际，和 `torch==2.0.0` 的组合有明显兼容性风险；通常应按 torchdata 发布代际选择与 torch 2.0 对应的版本，而不是保留 0.8.0。由于代码不使用它，最稳妥的复现判断是：不要把 torchdata 作为本项目核心依赖，也不要为了它升级 PyTorch。

### DGL 2.1.0

DGL 没有出现在任何 Python import 中。项目自己的 `model/layer_utils.py` 和 `model/bipartite_transformer.py` 实现了邻居索引、消息聚合和 attention，没有调用 DGL graph API。因此无法以当前仓库证明 DGL 2.1.0 是必要依赖，也没有必要为了 DGL 改动 PyTorch 2.0 版本。

### DeepChem 2.8.0 and RDKit 2022.9.5

这两个包是代码真实使用的分子处理依赖。代码只使用 DeepChem 的 `RDKitDescriptors`，并大量直接使用 RDKit 的 SMILES、构象、BRICS、Murcko、SMARTS 和绘图 API。当前调用形式与给定版本匹配；为论文复现保留 `DeepChem 2.8.0`、`RDKit 2022.9.5` 比追新更合理。其二进制包在 Windows 上的可获取性需按实际 Python/平台渠道验证，不能把 Linux 导出构建字符串原样搬运。

### NumPy, Pandas, SciPy, scikit-learn, Matplotlib

`numpy==1.24.2`、`pandas==1.3.5`、`scikit-learn==1.2.2`、`matplotlib==3.7.1` 满足代码中的直接 API。SciPy 虽在 yml 中，但源码不直接使用，不能据此要求升级或降级。未发现这些版本之间由当前代码触发的明显 API 冲突。

## 6. Platform-specific Configuration

### 不能在 Windows 原样照搬的内容

- `ld_impl_linux-64`：Linux x86_64 的 Conda linker 包。
- `libgcc-ng`、`libgomp`、`libstdcxx-ng`：Linux GNU runtime/OpenMP/C++ runtime。
- 带 `linux-64` 的 build 字符串，以及部分 `h06a4308`、`h7b6447c` 等 Linux 构建产物。
- `prefix: /root/miniconda3`：作者机器的绝对安装路径，不是项目依赖；在 Windows 上没有对应目录语义。
- `name: base`：作者把环境导出自 Conda base；不代表项目必须运行在 base 环境。

### 可以让 Conda 自动重新解析的内容

`python=3.8.10`、`numpy=1.24.2`、`pandas=1.3.5` 等版本约束可以保留版本号、去掉 Linux build 字符串，由 Conda 按 Windows 平台重新选择 build。`ca-certificates`、`openssl`、`sqlite`、`tk`、`zlib`、`setuptools` 等基础包也应由 Conda 重新求解，而不是复制作者机器的精确 build。

### 应保持的复现版本

在目标平台有可用构建的前提下，建议保持 Python 3.8.10、PyTorch 2.0.0、CUDA 11.8、torchvision 0.15.1、DeepChem 2.8.0、RDKit 2022.9.5、NumPy 1.24.2、Pandas 1.3.5、scikit-learn 1.2.2 和 Matplotlib 3.7.1。版本变更必须有安装可得性或代码/API 证据；当前核验没有理由为了“更新”而升级这些包。

## 7. Missing Dependencies

| 包名 | 使用位置 | 用途 | 可能由谁间接安装 | 建议 |
|---|---|---|---|---|
| `hyperopt` | `run_main.py` 第 1、2、4 行 | 超参数搜索对象和函数导入；即使当前搜索逻辑主要用 `product`，模块顶层 import 仍会执行 | 当前文件没有可可靠依赖其间接安装；`wandb`、PyTorch 等不会提供它 | **建议显式加入 environment.yml**；这是当前源码级缺失的直接依赖 |

没有发现其他“代码直接 import、environment.yml 完全没有对应包”的第三方包。`Electronegativity.pkl` 和 pickle 数据缓存是文件输入，不属于可用 pip/Conda 依赖，需单独准备。

## 8. Unused / Potentially Unnecessary Dependencies

重点未使用项如下，保留与否不在本次修改范围内：

1. 框架类：`torchvision`、`torchdata`、`dgl`。
2. 降维/可视化类：`umap`、`umap-learn`、`seaborn`、Jupyter/Notebook 全套。
3. 实验跟踪类：`wandb`、`tensorboard`、`tensorboard-data-server`、`tensorboard-plugin-wit`。
4. 服务、认证和 notebook 运行周边：`aiofiles`、`aiosqlite`、`argon2-*`、`google-auth*`、`jupyter-*`、`prometheus-client`、`supervisor`、`docker-pycreds`、`sentry-sdk` 等。
5. 源码没有直接 import 的科学计算周边：`scipy`、`joblib`、`networkx`、`pubchempy`、`requests`、`pillow`、`py4j` 等。

这是“当前仓库代码没有明显使用”的清单，不是删除建议。尤其是间接依赖不能只按 import 文本机械删除。

## 9. Required Changes

### 必须修改

1. 为 Windows 创建环境时去掉 `prefix: /root/miniconda3`。
2. 去掉或让 Conda 重新解析 Linux 专属包和 build 信息，至少不能保留 `ld_impl_linux-64`、`libgcc-ng`、`libgomp`、`libstdcxx-ng` 这些 Linux 绑定项。
3. 将 `hyperopt` 明确加入环境文件，或在安装环境后明确执行与 yml 配套的 `pip install hyperopt`；否则 `run_main.py` 会在 import 阶段失败。
4. 明确 PyTorch CUDA 11.8 wheel 的安装来源，并在目标 Windows 机器验证 `torch==2.0.0+cu118` 和 `torchvision==0.15.1+cu118` 能够同时安装。仅写 pip 版本号不足以保证干净环境可创建。
5. 若目标是“项目可运行”而不只是“依赖可安装”，还必须处理仓库本身缺失/不一致的入口问题：不存在的模型模块、未定义的 `scaffold_split`、未定义的 motif 配置属性，以及当前 README 所述但工作区没有的 pickle 数据。这些不应通过盲目升级依赖解决。

### 建议修改

1. 将 `name: base` 改为项目专用环境名，避免污染作者或用户的 Conda base；这不是包兼容性硬错误，但更利于复现。
2. 保留 Python 3.8.10 和已确认的分子/模型版本，删除精确 Linux build 字符串后重新求解平台构建。
3. 对 `torchdata==0.8.0` 做二选一：当前项目不使用时从项目核心环境中移除；若某个未提交的外部脚本确实需要它，则按该脚本和 PyTorch 2.0 的兼容矩阵锁定对应版本，不要直接采用 0.8.0。
4. `dgl`、`torchvision`、UMAP、wandb、TensorBoard 和 Jupyter 周边可作为作者工具环境保留，但不应在项目运行依赖审查中宣称为 HMG-MRL 核心必需项。
5. 为论文复现记录 PyTorch index、GPU 驱动/CUDA runtime、操作系统和实际安装结果；这些信息当前 yml 没有完整表达。

### 可以保持

在平台允许且能成功解析的前提下，可以保持：Python 3.8.10、PyTorch 2.0.0、CUDA 11.8、torchvision 0.15.1、DeepChem 2.8.0、RDKit 2022.9.5、NumPy 1.24.2、Pandas 1.3.5、SciPy 1.10.1、scikit-learn 1.2.2、Matplotlib 3.7.1 和 tqdm 4.61.2。当前没有代码证据要求升级它们。

## 10. Recommended Installation Strategy

推荐按以下顺序处理，重点是保持论文复现版本而不是全面升级：

1. 复制 `environment.yml` 作为安装草稿，移除 `prefix`、Linux 专属 Conda 包和 Linux build 字符串；保留版本号和项目真正需要的核心包。
2. 使用项目专用环境名，固定 Python 3.8.10。让 Conda 针对 Windows 重新选择 `openssl`、runtime、SQLite、压缩库等平台包。
3. 先安装 PyTorch 2.0.0 + CUDA 11.8 及 torchvision 0.15.1 的同源 wheel，并验证 `import torch; torch.cuda.is_available()`。如果目标机没有 NVIDIA GPU，不应把 `+cu118` 环境当作可运行 CPU 环境。
4. 安装/验证 DeepChem 2.8.0、RDKit 2022.9.5、NumPy 1.24.2、Pandas 1.3.5、scikit-learn 1.2.2、Matplotlib 3.7.1、tqdm 4.61.2，并显式安装 `hyperopt`。
5. 只在外部代码确实需要时再安装/保留 DGL、torchdata、torchvision、wandb、TensorBoard、UMAP 等非当前路径依赖；不要为了修复一个未使用包升级 PyTorch。
6. 做最小导入检查：`config`、`other_utils`、`preprocess.Featurizer_atom_bond`、`preprocess.get_atom_bond_frag_info`、`model.bipartite_transformer` 和 `run_main`。导入检查之后，再准备 `data/*.pickle` 并修复仓库自身的入口缺失问题。

最终判断仍是：**当前 `environment.yml` 不能直接作为 Windows 上当前 HMG-MRL 项目的实际运行环境。** 修改平台绑定信息、补齐 `hyperopt`、明确 PyTorch wheel 来源后，依赖层面才具备可验证的创建条件；但当前仓库代码自身仍有独立的运行阻断，环境核验不能把这些问题掩盖成“安装成功即可运行”。