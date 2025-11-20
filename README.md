

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td width="30%">
    <p align="center">
      <img src="logo.jpg" alt="TxConformal logo" width="220">
      </p>
    </td>
    <td> 
      <h1>Controlling False Discoveries <br> in AI-Driven Therapeutic Discovery</h1>  
    </td>
  </tr>
</table>



This repository hosts softwares and reproduction codes for the paper:

**TxConformal: Controlling False Discoveries in AI-Driven Therapeutic Discovery**

Ying Jin*, Kexin Huang*, Nathaniel Diamant, Kerry R. Buchholz, Steven T. Rutherford, Nicholas Skelton, Tommaso Biancalani, Gabriele Scalia, Jure Leskovec, and Emmanuel J. Candès
 
---

## 1. Quick start & general usage

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Quick-start example
```python
from txconformal import TxConformal
from txconformal.features import FeaturesProvider

# Step 1: feature builder with calibration/test data inputs
# f_cal/f_test as predicted values and emb_cal and emb_test embedding matrices
prov = FeaturesProvider(f_calib=f_cal, f_test=f_test,
                        E_calib=emb_cal, E_test=emb_test)
prov.prepare()                     # using default feature setup for TxConformal

# Step 2: TxConformal: build weights, p-values, and selection
txc = TxConformal(score_name="clip", cutoff=0.5)   # cutoff specifies Y_test > 0.5 as meaningful discovery
txc.fit(prov, y_calib, print_level=-1)   # suppresses EB logs
res = txc.select(method="bh", alpha=0.1)            # BH selection for FDR control

# Inspect results
print("selected indices:", res.idx)
print("threshold:", res.threshold)
```
---

## 2. Demo notebooks 

| Scenario | Notebook | Content |
| --- | --- | --- |
| ADMET / general tasks | `examples/general_tasks.ipynb` | Example usage for other selection scenarios in ADMET dataset, including FDR control, maximum false positives, minimum true positives, and FDP estimation. |
| Protein stability | `examples/protein_stability.ipynb` | Example usage for protein stability prediction task (regression problem) with FDR control. | 
| Enamine HTS screening | `examples/enamine.ipynb` | Reproduces the HTS Enamine prospective deployment (after diversity filtering): customized features, FDP estimates for top-ranked compounds. |

Each notebook presents required inputs (predictions, embeddings, cutoffs), how to configure `FeaturesProvider`, and how to interpret the TxConformal outputs for that scenario.

---

### 3. Advanced usage 

#### General workflow

- Step 1: prepare balancing features for calibration and test data via **FeaturesProvider**. 
  By default `prepare()` builds:
  - Soft block `phi = [f, f^2, pooled quantile bins, embeddings]`: forcing balance for PCs
  - Force block `[f, pooled bins]` (or fallback to `f` if embeddings are absent): forcing balance for these features
  - Backup block `[ [f, pooled bins], [f] ]`: fallback options when balancing soft and force blocks are not feasible
  
  > :bulb: *Customize* your balancing features via `.set_soft_block()` / `.set_force_block()` / `.set_backup_block()`. (See below)
- Step 2: Fit weights and construct p-values via **TxConformal**. 
  It asks for cutoff (so Y_test>cutoff is a true discovery), computes conformity scores, and converts them to weighted p-values.
- Step 3: Perform **Selection** by calling `.select()`. It supports several discovery tasks:
  - `method='bh'`: Benjamini-Hochberg procedure with FDR control below `alpha`
  - `method='fp_budget`: Selecting as many as possible while keeping false discoveries below `K`
  - `method='tp_min`: Selecting as few as possible with true positives above `K` (unless all selected)
  - `method='top_k`: Selecting top-`K` units with strongest p-values with FDP estimate


The output of `.select()` is a `SelectionResult` object containing:
- `idx`: indices of selected test units.
- `threshold`: threshold of selection on p-values.
- `fdp_est`: estimated FDP among selected units. 
- `meta`: collects meta-data, including 
  - `eb_meta`: successful entropy-balancing configuration (PCA bounds, tolerance, fallback info).
  - `score_meta`: which score function (`clip`, `residual`, etc.) and its hyperparameters.
  - `provider_meta`: dimensional summary of bins/embeddings produced during `prepare()`.

 



#### Customizing feature blocks

`FeaturesProvider.prepare` accepts `soft_mode`, `force_mode`, and `backup_mode` or users can override blocks after preparation with setters. This lets you tailor exactly which constraints entropy balancing must satisfy.

| Helper | Description |
| --- | --- |
| `get_soft_block()` | Returns `(phi_calib, phi_test)` used as “soft” features. |
| `get_force_block()` | Returns current mandatory constraint matrices. |
| `get_backup_block()` | Returns a list of fallback designs used if primary EB fails. |
| `set_soft_block(phi_c, phi_t)` | Replace default/cached soft balancing feature blocks. |
| `set_force_block(Fc, Ft)` | Replace default/cached forced balancing feature blocks (or set to `None`). |
| `set_backup_block([Fc1, ...], [Ft1, ...])` | Provide custom fallback stacks. |
| `get_block(mode="auto"/"f"/"f+bins"/"custom")` | Convenience builder for common patterns or custom callables. |
 

Example: use embeddings plus polynomial predictions in the soft block, but force balancing on bins only.
```python
prov.prepare(soft_mode="custom_phi", force_mode="f+bins")
prov.set_soft_block(custom_phi_c, custom_phi_t)
```

#### Controlling solver verbosity

`print_level` in `TxConformal.fit(...)` (forwarded to `retry_entropy_balancing`) controls how much the entropy-balancing solver prints:

- `print_level = -1` (default in examples) silences intermediate logs and numpy warnings.
- `print_level = 0` prints one line per retry attempt.
- Larger values show every EB iteration/residual—useful when debugging convergence issues.


