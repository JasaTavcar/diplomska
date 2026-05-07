import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

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

def load_parkinsons_data(device=None):
    file_path = "../data/parkinsons.data"
    df = pd.read_csv(file_path)
    df = df[:1000]
    target_cols = ['motor_UPDRS', 'total_UPDRS']
    feature_cols = df.columns.drop(['subject#', *target_cols])
    feature_names = feature_cols.tolist()
    X = df[feature_cols].values
    y = df['total_UPDRS'].values

    X = torch.tensor(X, dtype=torch.float64, device=device)
    y = torch.tensor(y, dtype=torch.float64, device=device).unsqueeze(1)
    return X, y, feature_names

def normalize_to_minus1_1(x, eps=1e-8):
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    return 2 * (x - x_min) / (x_max - x_min + eps) - 1

def load_car_data(device=None):
    path = "../data/car.data"

    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    df = pd.read_csv(path, sep=",", names=col_names)

    feature_names = col_names[:6]
    df_features = df[feature_names]

    # one-hot encode all categorical features
    df_encoded = pd.get_dummies(df_features, columns=feature_names)
    encoded_feature_names = df_encoded.columns.tolist()

    E_raw = torch.tensor(
        df_encoded.values,
        dtype=torch.float32,
        device=device
    )

    class_cat = df["class"].astype("category")
    C = torch.tensor(
        class_cat.cat.codes.values,
        dtype=torch.long,
        device=device
    )
    class_names = list(class_cat.cat.categories)

    instance_names = [str(i) for i in range(len(df))]

    return E_raw, C, instance_names, encoded_feature_names, class_names

def plot_projection(points, vectors, feature_names, class_codes, class_names, title, loading_cutoff=0.4, ax=None, text_scale=1.1):
    n = points.shape[0]
    if n > 100:
        rng = np.random.default_rng(0)
        keep = rng.choice(n, 100, replace=False)
        points = points[keep]
        class_codes = class_codes[keep]

    unique_classes = np.unique(class_codes)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    cmap = plt.get_cmap('tab10')

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for i, cls in enumerate(unique_classes):
        idx = class_codes == cls
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            color=cmap(i % 10),
            marker=markers[i % len(markers)],
            label=class_names[int(cls)],
            edgecolors='black',
            linewidths=0.7,
            alpha=0.9
        )

    circle = plt.Circle((0, 0), loading_cutoff, fill=False, linestyle='--', linewidth=0.8, color='gray')
    ax.add_patch(circle)

    for i, vec in enumerate(vectors):
        x, y = vec
        length = np.sqrt(x**2 + y**2)
        if length > loading_cutoff:
            ax.plot([0, x], [0, y], linestyle='-', linewidth=1.2, color='black')
            ax.text(
                x * text_scale, y * text_scale,
                feature_names[i],
                fontsize=8,
                ha='center',
                va='center'
            )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect('equal', 'box')

    ax.legend(title='Classes', loc='upper right', frameon=True, fontsize=9, title_fontsize=10)
    ax.set_title(title)

    return ax