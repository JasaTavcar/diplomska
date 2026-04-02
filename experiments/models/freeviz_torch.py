import numpy as np
from math import sin, cos, pi
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

    def train(self, E, C, tol=1e-2, patience=5, max_iter=2000, print_loss=True):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.05,
            momentum=0.9,
            nesterov=True,
        )

        history = []
        no_loss_decrease = 0
        prev_loss = None
        for _ in range(max_iter):
            P = self(E)
            loss = self.energy(P, C)
            history.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.normalize()

            if print_loss:
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

    def energy(self, P, C):
        F = torch.zeros_like(P)
        n_instances = P.shape[0]

        diff = P.unsqueeze(1) - P.unsqueeze(0)  # (N, N, 2)
        dist = torch.norm(diff, dim=2) + 1e-6   # (N, N)
        same = (C.unsqueeze(0) == C.unsqueeze(1))
        F_mag = torch.where(same, -dist, 1.0 / dist)
        F_vec = F_mag.unsqueeze(2) * (diff / dist.unsqueeze(2))
        F = F_vec.sum(dim=1)

        total_energy = torch.tensor(0.0, dtype=P.dtype, device=P.device)
        for e in range(n_instances):
            total_energy += torch.sqrt(F[e][0]**2 + F[e][1]**2)
        return total_energy

    def predict(self, E_ref, C_ref, E_query, k=5):
        P_ref = self(E_ref)
        P_query = self(E_query)
        k = min(k, P_ref.shape[0])

        diff = P_query.unsqueeze(1) - P_ref.unsqueeze(0)
        dist = torch.norm(diff, dim=2)
        nn_idx = torch.topk(dist, k, largest=False).indices

        classes, class_idx = torch.unique(C_ref, sorted=True, return_inverse=True)
        nn_class_idx = class_idx[nn_idx]
        votes = torch.zeros(P_query.shape[0], classes.numel(), dtype=torch.int64, device=dist.device)
        votes.scatter_add_(1, nn_class_idx, torch.ones_like(nn_class_idx, dtype=torch.int64))
        return classes[votes.argmax(dim=1)]

    def accuracy(self, E_ref, C_ref, E_query, C_query, k=5):
        preds = self.predict(E_ref, C_ref, E_query, k=k)
        return (preds == C_query).float().mean()