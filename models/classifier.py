import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class Classifier:
    def __init__(self, W, b, lr=0.01):
        self.W = W
        self.b = b
        self.lr = lr
        self.optimizer = optim.SGD([self.W, self.b], lr=self.lr)

    def train(self, Z, Y, epochs=10, patience=5, min_delta=1e-4):
        self.W.requires_grad_()
        self.b.requires_grad_()

        self.W.grad = None
        self.b.grad = None



        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=patience, min_lr=1e-6)

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            logits = Z @ self.W + self.b

            loss = F.cross_entropy(logits, Y)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            scheduler.step(loss)

            # Early stopping check
            if loss.item() < best_loss - min_delta:
                best_loss = loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    #print(f"Early stopping after {epoch+1} epochs.")
                    break

        return best_loss



    def predict(self, Z):
        logits = Z @ self.W + self.b
        return torch.argmax(logits, dim=1)
