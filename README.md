# HMG-MRL

Core implementation of **HMG-MRL: Hierarchical Multi-Granularity Molecular Representation Learning**.

This repository provides the core model, atom and bond featurization, motif extraction, and molecular graph construction code. The complete data-cleaning and end-to-end preprocessing notebook is not included in this public release.

---

## Requirements

The environment configuration required to run this project can be found in the `environment.yml` file.

---

## Files

### data

We use nine benchmark datasets for molecular property prediction, including six classification datasets (BACE, BBBP, SIDER, ClinTox, HIV, and Tox21) and three regression datasets (ESOL, FreeSolv, and Lipophilicity).

All datasets are obtained from [MoleculeNet](https://moleculenet.org/datasets-1).

---

### preprocess

- **Electronegativity.pkl**  
  Stores the Pauling electronegativity values of all atoms, used for extracting bond features based on electronegativity differences.

- **fg_dicts.txt**  
  Contains various functional groups and their corresponding SMARTS patterns for functional group identification.

- **Featurizer_atom_bond.py**  
  Extracts initial atom and bond features based on RDKit.

- **motif_utils.py**  
  Extracts diverse molecular motifs using three substructure partition strategies: BRICS fragmentation, scaffold decomposition, and functional group matching.

- **get_atom_bond_frag_info.py**  
  Models molecules as graph-structured data and extracts atom-, bond-, and motif-level feature matrices, molecular descriptors, neighbor index matrices, and RDKit index lists.

---

### model

- **act_func.py**  
  An activation function registry for flexible selection across different model layers.

- **bipartite_transformer.py**  
  Provides a complete implementation of the proposed HMG-MRL model.

- **layer_utils.py**  
  A utility library for reusable low-level neural network components.

---

### Root-level code files

- **other_utils.py**  
  Provides auxiliary training utilities, initialization, and random seed control. The scaffold-splitting wrapper requires a separately supplied `scaffold_split` implementation.

- **config.py**  
  Implements a `Config` class for centralized management of all project parameters, including model hyperparameters and training settings.

- **run_main.py**  
  Contains reference classification training and evaluation code that consumes prepared feature dictionaries and label tables. It is not a standalone end-to-end reproduction entry point.

---

## Public Code Scope

- `model/` implements the three-granularity model, cross-granularity communication, alignment loss, and attention-based fusion.
- Atom embeddings from bipartite message passing are used directly for readout and cross-granularity communication, without an intermediate bidirectional GRU.
- Each Motif Transformer layer applies LayerNorm after the residual addition in both its self-attention and feed-forward sublayers (Add & Norm).
- `preprocess/` provides atom/bond feature extraction, BRICS/Murcko/SMARTS motif extraction, motif feature aggregation, and graph construction utilities. Conformer, descriptor, and feature-cache utilities remain included.
- Dataset-specific cleaning and the notebook that orchestrates the complete preprocessing workflow are not distributed.

The model consumes prepared atom, bond, and motif feature tensors; atom/bond neighbor indices; validity masks; and molecular descriptors. See `GNN_atom_bond.forward` in `model/bipartite_transformer.py` for the input interface and `get_smiles_array` in `preprocess/get_atom_bond_frag_info.py` for the feature-dictionary adapter.

The reference training script expects externally prepared `data/<task>.pickle` feature dictionaries and `data/<task>_remained_df.pickle` label tables. The public files do not constitute a turnkey training pipeline; integration of the reference script with the user's preprocessing and training environment is required.
