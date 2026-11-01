# Training Decision Classifier — Single Sigmoid Neuron

**Python · NumPy · Matplotlib · Binary classification · Gradient descent**

An educational implementation of a **single sigmoid neuron** for learning a simple binary decision rule: *Should I attend a scheduled workout?* The three inputs are binary: whether a workout is scheduled, whether clean sportswear is available, and whether there is motivation to go.

This repository was originally named `train_perceptrone`. The historical `src/perceptone.py` and `SimplePerceptone` spellings are retained for compatibility.

> **Terminology:** despite the repository's original name, a differentiable sigmoid unit trained using binary cross-entropy is best described as a **logistic-regression / single-neuron classifier**, rather than the classical hard-threshold perceptron. This is **not** a multilayer perceptron (MLP).

## Classification rule

Attend only when a workout is scheduled **and** at least one of clean sportswear or motivation is available:

```text
attend = scheduled AND (sportswear_available OR motivated)
```

| Scheduled | Sportswear | Motivated | Expected |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 |

The training set contains all eight possible combinations — no duplicates or inconsistent labels.

## Quick start

Use Python 3.10+:

```bash
git clone https://github.com/marinaerokhina/train_perceptrone.git
cd train_perceptrone
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux), then:

```bash
python -m pip install -r requirements.txt
python -m src.start --no-plot
```

To display the learning curve, run `python -m src.start`. To export a chart without opening a window:

```bash
python -m src.start --no-plot --save-plot loss.png
```

## How learning works

1. Initialize three small weights and one bias with a deterministic seed.
2. Compute `p = sigmoid(X @ weights + bias)`.
3. Measure **binary cross-entropy** loss.
4. Update weights using the batch gradient `X.T @ (p - y) / n`.
5. Compare predictions against the complete eight-row truth table.

The model is deliberately simple and fully implemented with NumPy. No ML framework or pretrained model is used. Running the example shows per-row decisions, loss before and after training, and accuracy on **these same eight training examples**.

**Important:** Reporting 100% accuracy on a truth table used for training is an educational demonstration, **not** evidence of real-world predictive performance or generalization. The data is synthetic and deterministic.

## Repository structure

```text
src/
  perceptone.py      Sigmoid model, binary cross-entropy and batch updates
  start.py           Complete truth table, training and optional graph
  first.py           Original early exploration (preserved unchanged)
tests/
  test_perceptone.py Unit tests for rule, training and input validation
README.md            English documentation
README_RUS.md        Original detailed Russian notes
requirements.txt     Required Python packages
```

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions checks Python imports, executes unit tests and runs the headless demonstration. The graph is optional, keeping CI compatible with headless systems.

## Follow-up ideas

- Compare a classical step-function perceptron with sigmoid+BCE optimization
- Explain linear separability with an interactive plot
- Explore a nonlinearly separable dataset and why a multilayer network becomes necessary

**Author:** Marina Erokhina · Small independent educational machine-learning project.
