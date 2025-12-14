<<<<<<< HEAD
# ML From Scratch

Machine learning algorithms implemented **from scratch** using NumPy, with a focus on understanding the **mathematical foundations** and **core mechanics** behind each model.

This repository is designed as a **learning-focused codebase**. Implementations are intentionally explicit and low-level to make the underlying ideas easy to follow.

---

## Purpose

- Understand how core ML algorithms work internally
- Practice matrix calculus and gradient-based optimization
- Build solid fundamentals before using high-level frameworks
- Develop clean, modular, and readable ML code

This repo is intentionally **low-level**.

---

## Implemented Algorithms

### 1. Linear Regression

**From scratch implementation including:**

* Cost function (MSE)
* Gradient descent
* Learning rate experiments
* Loss convergence visualization

 `linear_regression/`

---

### 2. Multivariate Linear Regression

* Vectorized implementation
* Feature scaling
* Matrix-based gradient descent

 `multivariate_linear_regression/`

---

### 3. Logistic Regression

* Sigmoid activation
* Binary cross-entropy loss
* Gradient descent optimization
* Decision boundary visualization

 `logistic_regression/`

---

### 4. Neural Network (Forward Propagation Only)

* Simple feedforward neural network implemented from scratch
* Dense layer abstraction using matrix multiplication
* Sigmoid activation function
* Manual sequential forward pass (no training)
* Minimal test case with random weights

! This implementation focuses on **architecture and forward propagation** only.
Planned extensions are outlined in the Next Steps section.
`neural_network/`

---

### 5. Regularization

* L2 regularization
* Regularized gradients

`regularization/`

---

### 6. Utilities
Reusable helper functions:
- Dataset splitting
- Feature standardization
- Evaluation metrics (accuracy, precision, recall, F1)

 `utils/`

---
## Tech Stack

* Python
* NumPy
* Matplotlib
* No external ML libraries used (pure NumPy implementations)

---

## How to Run

```bash
pip install numpy matplotlib
python linear_regression/linear_regression_from_scratch.py
```

---

## Author

Baran Onal
Computer Engineering Student

---

## Next Steps

* Extend neural network depth
* Publish an **advanced neural network implementation** as a separate repository and link it here

---

For reviewers: this repository documents my learning progress through clear, from-scratch implementations, and each module represents concepts whose underlying mathematics I can explain at a foundational level.
=======
# ml-from-scratch
Machine learning algorithms implemented from scratch with NumPy.
>>>>>>> d24e66f5a7d44aa5445f73bea2a5b1beb5a41542
