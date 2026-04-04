import pandas as pd
import torch

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

def scale_features(E):
    E_min = E.min(dim=0, keepdim=True).values
    E_max = E.max(dim=0, keepdim=True).values
    E_scaled = (E - E_min) / (E_max - E_min)
    return E_scaled

def load_zoo_data(device=None):
    path = "../data/zoo.tab"

    df = pd.read_csv(path, sep="\t", skiprows=[1, 2])

    instance_names = df["name"].tolist()
    feature_cols = [col for col in df.columns if col not in ("name", "type")]
    feature_names = feature_cols
    E_raw = torch.tensor(
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

    return E_raw, C, instance_names, feature_names, class_names