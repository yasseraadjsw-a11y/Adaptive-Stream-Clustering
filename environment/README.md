# Environment

Supported runtime: **Python 3.11**.

Install with either:

```bash
python -m pip install -r requirements.txt
```

or:

```bash
conda env create -f environment/environment.yml
conda activate adaptive-stream-clustering
```

The environment explicitly includes NumPy, SciPy, scikit-learn, Matplotlib, pandas, psutil, Numba and pytest. Numba is declared so clean installations follow the tested accelerated paths rather than silently depending on a machine-specific preinstallation.
