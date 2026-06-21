[README.md]
# GEBSW

This repository contains the main experimental code for the paper:

**Generalized Energy-based Sliced Wasserstein Distance**

GEBSW unifies nonlinear projection functions and energy-based adaptive slice weighting for distribution comparison and point-cloud reconstruction.

## Files

| File | Description |
|---|---|
| `PCR-ModelNet40.py` | Point-cloud reconstruction experiments on ModelNet40. |
| `PCR-ShapeNet.py` | Point-cloud reconstruction experiments on ShapeNet. |
| `gebsw_w2_fidelity.py` | Wasserstein-fidelity experiments on controlled distribution pairs. |
| `log-computational.py` | Runtime scaling experiments for the empirical GEBSW estimator. |
| `dragon_bunny_gebsw_mechanis.py` | Mechanism visualization for SW, GSW, EBSW, and GEBSW. |

## Requirements

The code was developed with Python 3 and requires the following packages:

```bash
pip install numpy scipy pandas matplotlib torch trimesh h5py tqdm statsmodels
