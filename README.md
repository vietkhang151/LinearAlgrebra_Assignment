# Linear Algebra Assignment: Image Compression via Householder QR Decomposition

This repository contains the implementation of the **Householder QR Decomposition** algorithm built from scratch using NumPy, with practical applications in low-rank matrix approximation and digital image compression.

## Project Overview
QR decomposition is a fundamental operation in numerical linear algebra where a matrix $A$ is factored into an orthogonal matrix $Q$ and an upper triangular matrix $R$. While libraries like `scipy` or `numpy.linalg` offer built-in functions, this project focuses on tracking and implementing the algorithm from first principles using **Householder Reflections** to ensure maximum numerical stability.

### Key Applications Included:
1. **Small-Scale Verification:** Validating the orthogonal and upper-triangular properties on a small $4 \times 3$ matrix.
2. **Real Image Compression:** Compressing a grayscale image (`B2DBy.jpg`) using rank-$k$ approximations and analyzing the reconstruction error under the Frobenius norm.
3. **Synthetic Low-Rank Matrix Analysis:** Testing the algorithm on a large-scale controlled low-rank matrix ($256 \times 256$, Rank=20) to observe how data can be perfectly reconstructed once $k$ reaches the true rank.

---

## Mathematical Background

### 1. Householder Reflection
A Householder reflection is an orthogonal transformation represented by a matrix $H$:
$$H = I - 2vv^T$$
where $v$ is a unit vector orthogonal to the hyperplane of reflection. Multiplying a vector $x$ by $H$ reflects $x$ across this hyperplane.

### 2. QR Algorithm Steps
To introduce zeros below the diagonal of the $k$-th column of matrix $A$:
1. Isolate the subvector $x = A[k:, k]$.
2. Compute its Euclidean norm: $\alpha = \pm \|x\|_2$ (the sign is chosen opposite to $x_0$ to avoid numerical cancellation).
3. Construct the Householder vector: $u = x - \alpha e_1$, and normalize it: $v = \frac{u}{\|u\|_2}$.
4. Form the local reflection matrix: $H_{\text{local}} = I - 2vv^T$.
5. Embed $H_{\text{local}}$ into the global identity matrix to get $H_k$, then update $R = H_k R$ and $Q = Q H_k$.

### 3. Low-Rank Approximation
After computing $A = QR$, an approximate matrix $A_k$ of rank $k$ is constructed by keeping only the first $k$ columns of $Q$ and the first $k$ rows of $R$:
$$A_k = Q[:, :k] \times R[:k, :]$$

---

## Code Structure & Functions

The main execution script is `qr_householder.py`. Below are its core functions:

* `householder_qr(A)`: Implements the sequential Householder reflections to compute $Q$ and $R$.
* `low_rank_approximation(Q, R, k)`: Computes the truncated matrix product $Q_k R_k$ for a given rank $k$.
* `calculate_error(A, Ak)`: Computes the relative error using the Frobenius norm:
  $$\epsilon_k = \frac{\|A - A_k\|_F}{\|A\|_F}$$
* `experiment()`: Runs a small $4 \times 3$ test to print intermediate matrix states.
* `experiment_1()`: Processes the grayscale image `B2DBy.jpg`, displays the original vs. compressed images across different values of $k$, and plots the relative error curve.
* `experiment_2()`: Validates exact rank recovery on a $256 \times 256$ matrix with a controlled mathematical rank of 20.

---

## Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries. You can install the dependencies via `pip`:

```bash
pip install numpy matplotlib pillow
