import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# Construct a test dataset
n = 10
x1 = np.random.randn(n) * 5 # High variance, not predictive
x2 = np.linspace(0, 10, n) # Highly predictive
x3 = np.random.randn(n) * 0.1 # Low variance noise
X = np.vstack([x1, x2, x3]).T

# Y depends mostly on x2 + small noise
Y = 10 * x2 + np.random.randn(n) * 0.5

def viz_3d(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', s=50)
    for i in range(len(X)):
        ax.text(X[i, 0], X[i, 1], X[i, 2], f"P{i}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    plt.show()

def viz_2d(X):
    plt.scatter(X[:, 0], X[:, 1])
    for i in range(len(X)):
        plt.text(X[i, 0], X[i, 1], f"P{i}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def pca_projection(X, k=2):
    X_centered = X - X.mean(dim=0)
    U, S, Vt = torch.linalg.svd(X_centered)
    return X_centered @ Vt[:k].T

class SupervisedPCA(torch.nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.L = torch.nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float64))

    def project(self, X, Y):
        XL = X @ self.L
        return XL @ torch.linalg.pinv(XL) @ Y
    
def train(model, X, Y, lam, steps=200):
    # Initialize with PCA so rank condition holds
    with torch.no_grad(): # don't break autograd
        model.L.copy_(init_L_pca(X, k=model.L.shape[1]))

    history = []

    for step in range(steps):

        # 1. Euclidean gradient
        grad_euc = euclidean_grad(X, Y, model.L, lam)

        # 2. Grassmann projection
        grad = grassmann_grad(model.L, grad_euc)

        # 3. Armijo step size
        eta = armijo(X, Y, model.L, grad, lam)

        # 4. Geodesic update
        with torch.no_grad():
            model.L.copy_(geodesic_step(model.L, grad, eta))

        # Orthogonality sanity check
        I = torch.eye(model.L.shape[1], dtype=model.L.dtype, device=model.L.device)
        orthogonality_drift = torch.norm(model.L.T @ model.L - I)
        assert orthogonality_drift < 0.1, f"Orthogonality drift too high: {orthogonality_drift.item():.3f} on step {step}"
        
        loss = loss_fn(X, Y, model.L, lam)
        history.append(loss.item())

        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    return history

def train_lambda_unknown(model_class, X, Y, output_dim=2, steps=200):
    lambda_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_loss = float('inf')
    best_model = None

    for lam in lambda_candidates:
        # Initialize a fresh model for each lambda
        model = model_class(input_dim=X.shape[1], output_dim=output_dim)
        
        # Train L for this lambda
        train(model, X, Y, lam=lam, steps=steps)

        # Compute normalized loss
        pred_loss, recon_loss = normalized_loss(X, Y, model.L)
        norm_loss = pred_loss + recon_loss
        norm_loss_value = norm_loss.item()
        print(f"Lambda {lam}: normalized loss = {norm_loss_value:.4f}")

        if norm_loss_value < best_loss:
            best_loss = norm_loss_value
            best_lambda = lam
            best_model = model

    print(f"Selected lambda: {best_lambda} with normalized loss {best_loss:.4f}")
    return best_lambda, best_model


def armijo(X, Y, L, grad, lam, alpha=1.0, beta=0.2, c=1e-4):
    loss0 = loss_fn(X, Y, L, lam)
    grad_norm_sq = torch.norm(grad)**2

    max_iters = 50
    while max_iters > 0:
        max_iters -= 1
        L_new = geodesic_step(L, grad, alpha)
        loss_new = loss_fn(X, Y, L_new, lam)

        if loss_new <= loss0 - c * alpha * grad_norm_sq:
            break

        alpha *= beta

    if max_iters == 0:
        alpha = 1e-12  # fallback small step

    #print("alpha:", alpha)
    return alpha

def loss_fn(X, Y, L, lam=1.0):
    # X: (n, p)
    # L: (p, k)

    XL = X @ L # (n, k)

    # pseudoinverse
    XL_pinv = torch.linalg.pinv(XL, rtol=1e-6)

    # projection matrix onto span(XL)
    P = XL @ XL_pinv # (n, n)

    # prediction term
    pred_loss = torch.norm(Y - P @ Y)**2

    # reconstruction term
    X_recon = XL @ L.T
    pca_loss = torch.norm(X - X_recon)**2

    return pred_loss + lam * pca_loss

def normalized_loss(X, Y, L):
    XL = X @ L  # (n, k)
    XL_pinv = torch.linalg.pinv(XL, rtol=1e-6)
    P = XL @ XL_pinv  # projection onto XL
    pred_loss = torch.norm(Y - P @ Y)**2 / torch.norm(Y)**2

    X_recon = XL @ L.T
    recon_loss = torch.norm(X - X_recon)**2 / torch.norm(X)**2

    return pred_loss, recon_loss

def variance_explained(X, L):
    return torch.norm(X @ L, 'fro')**2 / torch.norm(X, 'fro')**2

def euclidean_grad(X, Y, L, lam):
    XL = X @ L # (n, k)
    XL_pinv = torch.linalg.pinv(XL, rtol=1e-6)

    P = XL @ XL_pinv # projection onto span(XL)
    P_perp = torch.eye(X.shape[0], device=X.device, dtype=X.dtype) - P

    term1 = -2 * (XL_pinv @ Y @ Y.T @ P_perp @ X) # (k, p)
    term1 = term1.T # match shape (p, k)

    term2 = -2 * lam * ((X.T @ X) @ L) # (p, k)

    return term1 + term2

def grassmann_grad(L, grad):
    return grad - L @ (L.T @ grad)

def geodesic_step(L, grad, step_size):
    # SVD of negative Riemannian gradient
    U, S, Vt = torch.linalg.svd(-grad, full_matrices=False)
    # U: (p, k), S: (k,), Vt: (k, k)

    # build diagonal sin/cos matrices
    cos_term = torch.diag(torch.cos(step_size * S))
    sin_term = torch.diag(torch.sin(step_size * S))

    # geodesic update
    L_new = (
        L @ Vt.T @ cos_term @ Vt
        + U @ sin_term @ Vt
    )

    return L_new

def init_L_pca(X, k):
    U, S, Vt = torch.linalg.svd(X)
    return Vt[:k].T   # (p, k)

def variation_explained(X, L):
    XL = X @ L          # (n, k)
    X_recon = XL @ L.T  # (n, p)
    varex = torch.norm(X_recon, 'fro')**2 / torch.norm(X, 'fro')**2
    return varex

def prediction_error(X, L, Y):
    XL = X @ L
    P = XL @ torch.linalg.pinv(XL, rtol=1e-6)
    mse = torch.mean((Y - P @ Y)**2)
    return mse

# simple manually constructed dataset
"""
viz_3d(X)
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)  # (n, 1)
X = X - X.mean(dim=0)
Y = Y - Y.mean(dim=0)
best_lambda, best_model = train_lambda_unknown(SupervisedPCA, X, Y, output_dim=1, steps=100)
L_final = best_model.L.detach().numpy()
feature_names = ["x1", "x2", "x3"]
for i in range(L_final.shape[1]):
    components = [f"{L_final[j, i]:.3f}*{feature_names[j]}" for j in range(3)]
    print(f"PC{i+1} = " + " + ".join(components))
"""

# proper dataset: residential buildings
file_path = "data/parkinsons.data"
df = pd.read_csv(file_path)
df = df[:1000]
print(df.head())
target_cols = ['motor_UPDRS', 'total_UPDRS']
feature_cols = df.columns.drop(['subject#', *target_cols])
X = df[feature_cols].values
y = df['total_UPDRS'].values

#X = df.iloc[:, :-2].values
#y = df.iloc[:, -1].values
X = torch.tensor(X, dtype=torch.float64)
y = torch.tensor(y, dtype=torch.float64).unsqueeze(1)

# Keep PCA objective in original feature scale: center only.
X = X - X.mean(dim=0)
y = y - y.mean(dim=0)
print(X.shape)
print(y.shape)

best_lambda, best_model = train_lambda_unknown(SupervisedPCA, X, y, output_dim=2, steps=200)
L_final = best_model.L.detach()
pred_loss, recon_loss = normalized_loss(X, y, L_final)
print(f"MSE: {prediction_error(X, L_final, y):.4f}")
print(f"Variance explained: {variance_explained(X, L_final):.4f}")