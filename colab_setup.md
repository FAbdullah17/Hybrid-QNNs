# Colab Setup Guide for Quantum Machine Learning Project

## 1. Clone the GitHub Repository
```python
!git clone https://github.com/<your-org-or-username>/<your-repo-name>.git
%cd <your-repo-name>
```

## 2. Install Dependencies
```python
!pip install -r requirements.txt
```

## 3. Run Experiments
- Edit experiment scripts in `experiments/` as needed.
- Run your assigned experiment:
```python
!python experiments/no_entanglement.py  # Asma
!python experiments/with_entanglement.py  # Farhan
!python experiments/varied_entanglement.py  # Fahad
```

## 4. Save Results and Plots
- Results and plots will be saved in `results/`.
- Download results from Colab or push changes to GitHub:
```python
# Download results
from google.colab import files
files.download('results/your_plot.png')

# Or push results to GitHub
!git add results/*
!git commit -m "Add new results and plots"
!git push
```

## 5. W&B Logging
- Make sure your W&B API key is set:
```python
import wandb
wandb.login()
```
- Metrics will sync automatically if integrated in your scripts.

---
**Each team member should use their own parameters/configs for efficient parallel experiments.**
