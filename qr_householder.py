import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Some useful functions and concepts:
# np.linalg.norm: Euclid distance of vector
# np.eye: Identity matrix
# np.zeros_like: Create an array of zeros with the same shape and type as a given array
# @: Matrix multiplication operator in Python
def householder_qr(A):
    m, n = A.shape
    R = A.astype(float).copy() #R  = Hk @ Hk-1 @ ... @ H1 @ A
    Q = np.eye(m)              #Q  = H1 @ H2 @ ... @ Hk-1 @ Hk

    # Iterate over each column to create Householder reflections
    for k in range(n):
        # Goal: find a Householder matrix H that will zero out the elements below the diagonal in the k-th column of R
        # 1. Vector x is the k-th column of R starting from row k
        x = R[k:, k].reshape(-1, 1)
        norm_x = np.linalg.norm(x)

        # 2. Alpha is chose based on the sign of the first element of x to avoid cancellation issues
        sign = -1 if x[0, 0] >= 0 else 1
        alpha = sign * norm_x

        # 3. Calculate e = v
        e1 = np.zeros_like(x)
        e1[0] = 1
        u = x - alpha * e1
        norm_u = np.linalg.norm(u)

        if norm_u > 1e-10:
            v = u / norm_u

            # 4. Householder local H_local = I - 2 * (v @ v.T)
            H_local = np.eye(len(x)) - 2 * (v @ v.T)

            # 5. Householder global H_k = identity with H_local at the bottom right
            H = np.eye(m)
            H[k:, k:] = H_local

            # 6. Apply to R and Q
            R = H @ R
            Q = Q @ H

    return Q, R

def low_rank_approximation(Q, R, k):
    """
    Compute rank-k approximation Ak = Qk * Rk.
    """
    return Q[:, :k] @ R[:k, :]

def calculate_error(A, Ak):
    """Calculate the relative Frobenius norm error"""
    return np.linalg.norm(A - Ak, 'fro') / np.linalg.norm(A, 'fro')


def print_matrix(name, matrix, precision=2):
    print(f"\n{name}:")
    for row in matrix:
        print("  " + "  ".join(f"{val:8.2f}" if abs(val) > 1e-10 else f"{0.0:8.2f}" for val in row))

# EXPERIMENT 1. SMALL SCALE VERIFICATION
def experiment_1():
    print("="*3)
    print("EXPERIMENT 1: SMALL SCALE (4x3)")
    A_small = np.array([
        [1, 2, 3],
        [-2, 1, -1],
        [2, -2, 1],
        [-4, 1, 2]
    ], dtype=float)
    Q_s, R_s = householder_qr(A_small)
    print_matrix("Original Matrix A", A_small)
    print_matrix("Orthogonal Matrix Q", Q_s)
    print_matrix("Upper Triangular R", R_s)
    print_matrix("Reconstructed A (Q * R)", Q_s @ R_s)
    print_matrix("Compressed Ak (k=1)", low_rank_approximation(Q_s, R_s, 1))

# EXPERIMENT 2: REAL IMAGE COMPRESSION
def experiment_2():
    print("\n" + "="*50)
    print("EXPERIMENT 2: REAL IMAGE COMPRESSION")
    print("="*50)

    image_path = 'B2DBy.jpg'
    # 1. Read and preprocess image
    print("Reading and preprocessing image...")
    img = Image.open(image_path).convert('L')
    img = img.resize((256, 256)) # Resize for moderate computational load
    A_img = np.array(img, dtype=float) / 255.0 # Normalize pixel values to [0, 1]
    # 2. QR Decomposition
    print("Performing Householder QR decomposition on image... (please wait)")
    Q_img, R_img = householder_qr(A_img)

    # 3. Compress and plot
    k_values = [10, 30, 60, 100]
    errors = []

    plt.figure(figsize=(15, 8))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(A_img, cmap='gray')
    plt.title(f"Original Image ({A_img.shape[0]}x{A_img.shape[1]})")
    plt.axis('off')

    print("Reconstructing compressed images...")
    for i, k in enumerate(k_values):
        Ak = low_rank_approximation(Q_img, R_img, k)
        error = calculate_error(A_img, Ak)
        errors.append(error)

        plt.subplot(2, 3, i + 2)
        plt.imshow(Ak, cmap='gray')
        plt.title(f"k = {k}\nError = {error:.4f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show() # Comparison image

    # DETAILED CALCULATIONS AND ERROR GRAPHING
    all_k = list(range(1, A_img.shape[1] + 1, 5))
    all_errors = [calculate_error(A_img, low_rank_approximation(Q_img, R_img, j)) for j in all_k]

    plt.figure(figsize=(10, 6))
    plt.plot(all_k, all_errors, color='red', marker='s', markevery=10, label=r'Error $\epsilon_k$')
    plt.title('The graph of the error approximates the Frobenius norm with respect to rank k.', fontsize=14)
    plt.xlabel('Rank k (Number of storage columns)', fontsize=12)
    plt.ylabel(r'Relative error $\epsilon_k$', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show() # Display the error graph

    # PRINT THE COMPRESSION EFFICIENCY ANALYSIS TABLE
    print(f"\n{'k':<5} | {'Error (%)':<12} | {'Compression ratio (%)':<15}")
    print("-" * 35)
    for k in [10, 50, 100, 150, 200]:
        error = calculate_error(A_img, low_rank_approximation(Q_img, R_img, k))
        # Formula: (k*(m+n)) / (m*n)
        compression_ratio = (k * (A_img.shape[0] + A_img.shape[1])) / (A_img.shape[0] * A_img.shape[1])
        saved = (1 - compression_ratio) * 100
        print(f"{k:<5} | {error*100:<12.2f} | {saved:<15.2f}")


# EXPERIMENT 3: LARGE-SCALE RANDOM MATRIX
def experiment_3():
    print("===SYNTHETIC LOW-RANK MATRIX (256x256, Rank=20)===")
    np.random.seed(42)
    B = np.random.randn(256, 20)
    C = np.random.randn(20, 256)
    A_large = B @ C

    Q_l, R_l = householder_qr(A_large)

    # Observe the area around point k=20
    k_values = [5, 10, 19, 20, 30]
    original_total = A_large.size

    print(f"{'Rank k':<10} | {'Elements Stored':<15} | {'Error':<20} | {'Saved (%)':<15}")
    print("-" * 65)

    for k in k_values:
        Ak = low_rank_approximation(Q_l, R_l, k)
        stored = (A_large.shape[0] * k) + (k * A_large.shape[1])
        error = calculate_error(A_large, Ak)
        saved = (1 - stored / original_total) * 100

        print(f"{k:<10} | {stored:<15} | {error:<20.4e} | {saved:.2f}%")


# MAIN EXECUTION
if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()