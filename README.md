# Kernel Methods & XOR Arbiter PUF Inversion

This project explores two machine learning problems: **semi-parametric regression using kernel methods** and **delay recovery in XOR Arbiter Physical Unclonable Functions (PUFs)**.

### Project Overview

**1. Semi-Parametric Regression (Hiring Prediction)**

- Developed a custom composite polynomial kernel for Kernel Ridge Regression to predict staffing requirements based on video length, popularity, and difficulty.
- Derived a valid Mercer kernel incorporating nonlinear feature interactions and a bias term.
- Performed extensive hyperparameter tuning, achieving an **R² score of 0.97004** with polynomial degree 3 and coefficient 0.16.

**2. XOR Arbiter PUF Inversion**

- Developed a mathematical approach to recover non-negative hardware delay parameters from a 1089-dimensional XOR Arbiter PUF model.
- Applied **Singular Value Decomposition (SVD)** for rank-one matrix factorization and non-negative least squares for delay reconstruction.
- Implemented recovery of 256 delay parameters across two 32-bit Arbiter PUFs.

### Tech Stack

Python | NumPy | Scikit-learn | Linear Algebra | Kernel Methods | SVD | Constrained Optimization

### Results

- **Regression:** R² = 0.97004
- **PUF Inversion:** Reconstructed eight 32-dimensional non-negative delay vectors from a 1089-dimensional model.

### Documentation

Refer to the project report for mathematical derivations, implementation details, and experimental results.
