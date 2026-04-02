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

    def train(self, E, C, model, tol=1e-3, patience=5, max_iter=2000):
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
            loss = self.energy(P, C)
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