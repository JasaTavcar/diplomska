import numpy as np

"""
Bair's method for SPCA - two phases:
1) compute regression coefficients and eliminate features with low coefficient
2) perform regular PCA on the remaining features
"""
class SPCA_Bair:
    def __init__(self, theta, n_components=1):
        self.theta = theta
        self.n_components = n_components
        self.selected_idx = None
        self.w = None
        self.gamma = None
        self.y_mean = None

    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape

        # Center y
        self.y_mean = y.mean()
        y_centered = y - self.y_mean

        # Step 1: compute scores s_j
        s = np.zeros(p)
        for j in range(p):
            x_j = X[:, j]
            norm = np.linalg.norm(x_j)
            if norm > 0:
                s[j] = x_j @ y_centered / norm

        # Step 2: select features
        self.selected_idx = np.where(np.abs(s) > self.theta)[0]
        if len(self.selected_idx) == 0:
            raise ValueError("No features selected. Try smaller theta.")
        X_theta = X[:, self.selected_idx]

        # Step 3: PCA via SVD
        _, _, Vt = np.linalg.svd(X_theta, full_matrices=False)
        self.w = Vt[: self.n_components, :] # (n_components, n_selected_features)

        # Step 4: project onto supervised PCs
        Z = X_theta @ self.w.T  # (n_samples, n_components)

        # Step 5: regression on projected components
        self.gamma, *_ = np.linalg.lstsq(Z, y_centered, rcond=None)

    def predict(self, X):
        X = np.asarray(X)
        X_theta = X[:, self.selected_idx]        
        Z = X_theta @ self.w.T
        y_pred = self.y_mean + Z @ self.gamma
        return y_pred

    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)