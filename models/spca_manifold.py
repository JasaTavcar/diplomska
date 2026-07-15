import numpy as np
import torch
import pandas as pd

class SupervisedPCA(torch.nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.L = torch.nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float64))
        self.loss_history = []
    
    def train(self, X, Y, lam, steps=200):
        self.to(device=X.device, dtype=X.dtype)
        
        # Initialize with PCA so rank condition holds
        with torch.no_grad(): # don't break autograd
            self.L.copy_(init_L_pca(X, k=self.L.shape[1]))

        for step in range(steps):

            # 1. Euclidean gradient
            grad_euc = euclidean_grad(X, Y, self.L, lam)

            # 2. Grassmann projection
            grad = riemann_grad(self.L, grad_euc)

            # 3. Armijo step size
            eta = armijo(X, Y, self.L, grad, lam)

            # 4. Geodesic update
            with torch.no_grad():
                self.L.copy_(geodesic_step(self.L, grad, eta))

            # Orthogonality sanity check
            I = torch.eye(self.L.shape[1], dtype=self.L.dtype, device=self.L.device)
            orthogonality_drift = torch.norm(self.L.T @ self.L - I)
            assert orthogonality_drift < 0.1, f"Orthogonality drift too high: {orthogonality_drift.item():.3f} on step {step}"
            
            loss = loss_fn(X, Y, self.L, lam)
            self.loss_history.append(loss.item())

            if step % 10 == 0:
                print(f"Step {step}: loss = {loss.item():.4f}")

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

def spca_normalized_loss(X, Y, L):
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

def riemann_grad(L, grad):
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