import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from math import sqrt, sin, cos, pi
import pandas as pd

import torch

class Freeviz(torch.nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        # Initialize on circle
        step = 2*pi/n_dim
        A_init = torch.stack([
            torch.tensor([sin(i * step), cos(i * step)])
            for i in range(n_dim)
        ])
        self.A = torch.nn.Parameter(A_init)  # (n_dim, 2)

    def normalize(self):
        with torch.no_grad():
            self.A -= self.A.mean(dim=0, keepdim=True) # Center
            norms = torch.norm(self.A, dim=1) # Scale so max norm = 1
            self.A /= norms.max()

    def forward(self, E):
        row_sums = E.sum(dim=1, keepdim=True)
        return (E @ self.A) / row_sums

# one iteration
def train(E, C, model, tol=1e-3, patience=5, max_iter=2000):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        nesterov=True,
    )

    history = []
    no_loss_decrease = 0
    prev_loss = None
    for _ in range(max_iter):
        P = model(E)
        loss = energy(P, C)
        history.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize()

        print("step:", len(history), "loss:", loss.item())
        current_loss = history[-1]
        if prev_loss is not None:
            rel_improvement = (prev_loss - current_loss) / abs(prev_loss)
            if rel_improvement < tol:
                no_loss_decrease += 1
            else:
                no_loss_decrease = 0
            if no_loss_decrease >= patience:
                break
        prev_loss = current_loss

def energy(P, C):
    F = torch.zeros_like(P)
    n_instances = P.shape[0]

    for e in range(n_instances):
        for f in range(e+1, n_instances):
            dx = P[e][0] - P[f][0]
            dy = P[e][1] - P[f][1]
            r = torch.sqrt(dx**2 + dy**2 + 1e-6)

            if C[e] == C[f]:
                F_ef = -r
            else:
                F_ef = 1/r

            F_efx = F_ef * dx/r
            F[e][0] += F_efx
            F[f][0] -= F_efx

            F_efy = F_ef * dy/r
            F[e][1] += F_efy
            F[f][1] -= F_efy

    total_energy = torch.tensor(0.0, dtype=P.dtype, device=P.device)
    for e in range(n_instances):
        total_energy += torch.sqrt(F[e][0]**2 + F[e][1]**2)
    return total_energy

def vizualize(P, C, A, feature_names, class_names, names):
    P_np = P.detach().cpu().numpy() if torch.is_tensor(P) else np.asarray(P)
    C_np = C.detach().cpu().numpy() if torch.is_tensor(C) else np.asarray(C)
    A_np = A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A)

    # scatter instance projections
    scatter = plt.scatter(P_np[:, 0], P_np[:, 1], c=C_np)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.gca().set_aspect('equal', 'box')

    # Plot each row of A as a vector from the origin
    for i, vec in enumerate(A_np):
        x, y = vec  # (A_i0, A_i1)
        if sqrt(x**2 + y**2) > 0.4:
            plt.plot([0, x], [0, y], linestyle='-', linewidth=2) 
            plt.text(x * 1.05, y * 1.05, feature_names[i], ha='left', va='bottom')

    # Enable hover tooltips
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        i = int(np.asarray(sel.index).item())
        i = max(0, min(i, len(C_np) - 1))
        cls = int(C_np[i])
        cls_name = class_names[cls] 
        sel.annotation.set_text(f"{names[i]}\n{cls_name}")
    plt.show()

def load_zoo_tab(path, device=None):
    df = pd.read_csv(path, sep="\t", skiprows=[1, 2])

    # --- metadata (keep as Python objects) ---
    instance_names = df["name"].tolist()
    feature_cols = [col for col in df.columns if col not in ("name", "type")]
    feature_names = feature_cols
    E = torch.tensor(
        df[feature_cols].values,
        dtype=torch.float32,
        device=device
    )
    types_cat = df["type"].astype("category")
    C = torch.tensor(
        types_cat.cat.codes.values,
        dtype=torch.long,
        device=device
    )
    class_names = list(types_cat.cat.categories)
    return E, C, instance_names, feature_names, class_names

def scale_features(E):
    E_min = E.min(dim=0, keepdim=True).values
    E_max = E.max(dim=0, keepdim=True).values
    E_scaled = (E - E_min) / (E_max - E_min)
    return E_scaled

if __name__ == "__main__":
    # Initialize matrices
    E_raw, C, names, feature_names, class_names = load_zoo_tab("data/zoo.tab")
    E = scale_features(E_raw)

    freeviz = Freeviz(E.shape[1]).to(E.device)
    train(E, C, freeviz)

    vizualize(freeviz(E), C, freeviz.A, feature_names, class_names, names)
