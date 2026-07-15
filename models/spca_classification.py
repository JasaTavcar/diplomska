import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from models.classifier import Classifier

class SPCA_Classification(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=2, num_classes=2, lr=0.01):
        super().__init__()
        self.L = torch.nn.Parameter(torch.randn(input_dim, latent_dim, dtype=torch.float64))
        self.W = torch.nn.Parameter(torch.randn(latent_dim, num_classes, dtype=torch.float64) * 0.01)
        self.b = torch.nn.Parameter(torch.zeros(num_classes, dtype=torch.float64))
        self.loss_history = []
        self.lr = lr  # learning rate for W and b updates

        # Initialize classifier trainer (will use self.W and self.b)
        self.classifier = Classifier(self.W, self.b, lr=self.lr)

    def classify_logits(self, X):
        Z = X @ self.L  # latent features
        logits = Z @ self.W + self.b
        return logits

    def forward(self, X):
        return self.classify_logits(X)
    
    def train(self, X, Y, lam, steps=200, classifier_epochs=50):
        self.to(device=X.device, dtype=X.dtype)
        
        # Initialize with PCA so rank condition holds
        with torch.no_grad(): # don't break autograd
            self.L.copy_(init_L_pca(X, k=self.L.shape[1]))

        for step in range(steps):
            # 1. Fix L, compute latent features Z
            Z = X @ self.L

            # 2. Train classifier W, b on Z
            classifier_loss = self.classifier.train(Z, Y, epochs=classifier_epochs)

            # 3. Compute gradients w.r.t. L with current classifier fixed
            grad_L = self.compute_L_grad(X, Y, lam)

            # 4. Grassmann projection on grad_L
            grad_L_proj = riemann_grad(self.L, grad_L)

            # 5. Armijo step size for L
            eta = armijo(X, Y, self.L, grad_L_proj, lam, self.W, self.b)

            # 6. Geodesic update for L
            with torch.no_grad():
                self.L.copy_(geodesic_step(self.L, grad_L_proj, eta))

            # Orthogonality sanity check
            I = torch.eye(self.L.shape[1], dtype=self.L.dtype, device=self.L.device)
            orthogonality_drift = torch.norm(self.L.T @ self.L - I)
            assert orthogonality_drift < 0.1, f"Orthogonality drift too high: {orthogonality_drift.item():.3f} on step {step}"
            
            # Log total loss for stats
            loss = loss_fn(X, Y, self.L, self.W, self.b, lam)
            self.loss_history.append(loss.item())

            if step % 10 == 0:
                print(f"Step {step}: loss = {loss.item():.4f}, classifier loss = {classifier_loss:.4f}")

    def compute_L_grad(self, X, Y, lam):
        # Compute gradient w.r.t L only, fixing W and b
        X = X.requires_grad_(False)
        Y = Y.requires_grad_(False)

        self.L.requires_grad_(True)
        self.W.requires_grad_(False)
        self.b.requires_grad_(False)

        Z = X @ self.L
        logits = Z @ self.W + self.b

        classification_loss = F.cross_entropy(logits, Y)
        X_recon = Z @ self.L.T
        pca_loss = torch.norm(X - X_recon)**2
        loss = classification_loss + lam * pca_loss

        loss.backward()
        grad_L = self.L.grad.clone()
        self.L.grad.zero_()
        self.L.requires_grad_(False)

        return grad_L




def armijo(X, Y, L, grad, lam, W, b, alpha=1.0, beta=0.2, c=1e-4):
    loss0 = loss_fn(X, Y, L, W, b, lam)
    grad_norm_sq = torch.norm(grad)**2

    max_iters = 50
    while max_iters > 0:
        max_iters -= 1
        L_new = geodesic_step(L, grad, alpha)
        loss_new = loss_fn(X, Y, L_new, W, b, lam)

        if loss_new <= loss0 - c * alpha * grad_norm_sq:
            break

        alpha *= beta

    if max_iters == 0:
        alpha = 1e-12  # fallback small step

    #print("alpha:", alpha)
    return alpha


# Update train calls armijo accordingly


def loss_fn(X, Y, L, W, b, lam=1.0):
    Z = X @ L
    logits = Z @ W + b
    classification_loss = F.cross_entropy(logits, Y)

    X_recon = Z @ L.T
    pca_loss = torch.norm(X - X_recon)**2

    return classification_loss + lam * pca_loss


def euclidean_grad(X, Y, L, W, b, lam):
    Z = X @ L
    logits = Z @ W + b

    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)
    # One-hot encode labels
    Y_onehot = torch.nn.functional.one_hot(Y, num_classes=logits.shape[1]).to(X.dtype)

    # Gradient w.r.t. logits
    grad_logits = probs - Y_onehot

    # Gradients on W and b
    grad_W = Z.T @ grad_logits
    grad_b = grad_logits.sum(0)

    # Gradient w.r.t Z
    grad_Z = grad_logits @ W.T

    # Gradient w.r.t L from Z
    grad_L_pred = X.T @ grad_Z

    # PCA regularization gradient
    grad_L_pca = 2 * lam * (X.T @ X) @ L

    # Total grad w.r.t L
    grad_L = grad_L_pred + grad_L_pca

    return grad_L, grad_W, grad_b


# Update train method to use new grads and manual steps for W, b
# Will implement next step

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