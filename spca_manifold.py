# 1: find 3D X and Y by hand that you know what it should converge to in 2D
#   - vizualize, try with regular PCA
# 2: construct boilerplate pytorch (with chatgpt)
# 3: define loss
# 4: implement full algorithm
# 5: use a better dataset 

import numpy as np
import torch
import matplotlib.pyplot as plt

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

# Supervised PCA
class SupervisedPCA(torch.nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.L = torch.nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float32))

    def project(self, X, Y):
        XL = X @ self.L
        return XL @ torch.linalg.pinv(XL) @ Y
    
# training loop
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
        #print("orthogonality drift:", torch.norm(model.L.T @ model.L - torch.eye(model.L.shape[0])))

        loss = loss_fn(X, Y, model.L, lam)
        history.append(loss.item())

        #if step % 20 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

    return history

def armijo(X, Y, L, grad, lam, alpha=1.0, beta=0.5, c=1e-4):
    loss0 = loss_fn(X, Y, L, lam)
    grad_norm_sq = torch.norm(grad)**2

    max_iters = 20
    while max_iters > 0:
        max_iters -= 1
        L_new = geodesic_step(L, grad, alpha)
        loss_new = loss_fn(X, Y, L_new, lam)

        if loss_new <= loss0 - c * alpha * grad_norm_sq:
            break

        alpha *= beta

    if max_iters == 0:
        alpha = 1e-9  # fallback small step

    #print("alpha:", alpha)
    return alpha

def loss_fn(X, Y, L, lam=1.0):
    # X: (n, p)
    # L: (p, k)

    XL = X @ L # (n, k)

    # pseudoinverse
    XL_pinv = torch.linalg.pinv(XL)

    # projection matrix onto span(XL)
    P = XL @ XL_pinv # (n, n)

    # prediction term
    pred_loss = torch.norm(Y - P @ Y)**2 # **2 -> Forbenious norm

    # reconstruction term
    X_recon = XL @ L.T
    pca_loss = torch.norm(X - X_recon)**2 # Forbenious

    return pred_loss + lam * pca_loss

def euclidean_grad(X, Y, L, lam):
    XL = X @ L # (n, k)
    XL_pinv = torch.linalg.pinv(XL)

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

#viz_3d(X)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)  # (n, 1)

# center matrices
X = X - X.mean(dim=0)
Y = Y - Y.mean(dim=0)

model = SupervisedPCA(input_dim=X.shape[1], output_dim=1)
history = train(model, X, Y, lam=100, steps=200)

L_final = model.L.detach().numpy()  # shape (3, k)
feature_names = ["x1", "x2", "x3"]

for i in range(L_final.shape[1]):
    components = [f"{L_final[j, i]:.3f}*{feature_names[j]}" for j in range(3)]
    print(f"PC{i+1} = " + " + ".join(components))