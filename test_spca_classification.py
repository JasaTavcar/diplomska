import torch
from models.spca_classification import SPCA_Classification

# Create random dataset
n_samples, n_features, n_classes = 10, 20, 2
X = torch.randn(n_samples, n_features, dtype=torch.float64)
Y = torch.randint(0, n_classes, (n_samples,), dtype=torch.long)

# Initialize model
model = SPCA_Classification(input_dim=n_features, latent_dim=2, num_classes=n_classes, lr=0.05)

# Print initial parameter stats
print(f"Initial W norm: {model.W.norm().item():.6f}")
print(f"Initial b norm: {model.b.norm().item():.6f}")

# Train model
model.train(X, Y, lam=0.1, steps=100, classifier_epochs=20)

# Print final parameter stats
print(f"Final W norm: {model.W.norm().item():.6f}")
print(f"Final b norm: {model.b.norm().item():.6f}")

# Print loss history with details
for i, loss in enumerate(model.loss_history):
    print(f"Step {i}: loss = {loss}")


