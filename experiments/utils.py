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

def select_lambda_classification(model_class, E, C, num_classes, lam_values=None,
                  test_size=0.2, steps=500, classifier_epochs=50, lr=0.05,
                  random_state=42):
    from sklearn.model_selection import train_test_split

    E_np = E.cpu().numpy()
    C_np = C.cpu().numpy()
    E_train, E_test, C_train, C_test = train_test_split(
        E_np, C_np, test_size=test_size, random_state=random_state
    )

    device = E.device
    E_train = torch.tensor(E_train, dtype=torch.float64, device=device)
    E_test = torch.tensor(E_test, dtype=torch.float64, device=device)
    C_train = torch.tensor(C_train, dtype=torch.long, device=device)
    C_test = torch.tensor(C_test, dtype=torch.long, device=device)

    train_mean = E_train.mean(dim=0, keepdim=True)
    E_train -= train_mean
    E_test -= train_mean

    if lam_values is None:
        lam_values = np.logspace(-4, 1, 12)

    var_explained = []
    accuracies = []

    for lam in lam_values:
        model = model_class(
            input_dim=E_train.shape[1], latent_dim=2,
            num_classes=num_classes, lr=lr
        ).to(device)
        model.train(E_train, C_train, lam=lam, steps=steps,
                    classifier_epochs=classifier_epochs)

        L = model.L.detach()

        ve = variation_explained(E_test, L)
        var_explained.append(ve.item())

        logits = model.classify_logits(E_test)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == C_test).float().mean()
        accuracies.append(acc.item())

        print(f"lambda = {lam:.6f}:  var_exp = {ve.item():.4f},  "
              f"accuracy = {acc.item():.4f}")

    sum_metric = [v + a for v, a in zip(var_explained, accuracies)]
    max_idx = int(np.argmax(sum_metric))
    best_lam = lam_values[max_idx]
    print(f"\nBest lambda = {best_lam:.6f} (sum = {sum_metric[max_idx]:.4f})")

    return best_lam


def select_theta_bair(X, y, theta_values=None, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from models.spca_bair import SPCA_Bair

    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if theta_values is None:
        theta_values = np.linspace(0.1, 5.0, 20)

    records = []

    for theta in theta_values:
        try:
            model = SPCA_Bair(theta=theta, n_components=1)
            model.fit(X_train, y_train)
            mse_valid = model.mse(X_valid, y_valid)
            L = np.zeros((X.shape[1], 1))
            L[model.selected_idx, 0] = model.w.ravel()
            Xv = torch.tensor(X_valid, dtype=torch.float64)
            L_t = torch.tensor(L, dtype=torch.float64)
            ve = variation_explained(Xv, L_t).item()
            print(f"theta = {theta:.3f}:  val_mse = {mse_valid:.4f},  val_ve = {ve:.4f}")
            records.append({"theta": theta, "mse": mse_valid, "ve": ve})
        except ValueError:
            print(f"theta = {theta:.3f}:  no features selected, skipping")

    if len(records) == 0:
        print("No theta succeeded, returning default.")
        return theta_values[0]

    mse_vals = np.array([r["mse"] for r in records])
    ve_vals = np.array([r["ve"] for r in records])
    eps = 1e-12
    mse_norm = (mse_vals - mse_vals.min()) / (mse_vals.max() - mse_vals.min() + eps)
    combined = ve_vals + (1 - mse_norm)
    best_idx = int(np.argmax(combined))
    best_theta = records[best_idx]["theta"]

    print(f"\nBest theta = {best_theta:.3f} (combined = {combined[best_idx]:.4f}, "
          f"ve = {records[best_idx]['ve']:.4f}, mse = {records[best_idx]['mse']:.4f})")
    return best_theta


def select_lambda_regression(X, y, lam_values=None, test_size=0.2, steps=200, random_state=42):
    from sklearn.model_selection import train_test_split
    from models.spca_manifold import SupervisedPCA

    device = X.device

    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state
    )

    X_train = torch.tensor(X_train, dtype=torch.float64, device=device)
    X_valid = torch.tensor(X_valid, dtype=torch.float64, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float64, device=device)
    y_valid = torch.tensor(y_valid, dtype=torch.float64, device=device)

    if lam_values is None:
        lam_values = np.logspace(-4, 1, 12)

    records = []

    for lam in lam_values:
        model = SupervisedPCA(input_dim=X_train.shape[1], output_dim=2)
        model.train(X_train, y_train, lam=lam, steps=steps)

        L = model.L.detach()
        pred_err = prediction_error(X_valid, L, y_valid)
        ve = variation_explained(X_valid, L).item()
        pred_err_val = pred_err.item()
        print(f"lambda = {lam:.6f}:  val_pred_err = {pred_err_val:.4f},  val_ve = {ve:.4f}")
        records.append({"lam": lam, "pred_err": pred_err_val, "ve": ve})

    pred_err_vals = np.array([r["pred_err"] for r in records])
    ve_vals = np.array([r["ve"] for r in records])
    eps = 1e-12
    pe_norm = (pred_err_vals - pred_err_vals.min()) / (pred_err_vals.max() - pred_err_vals.min() + eps)
    combined = ve_vals + (1 - pe_norm)
    best_idx = int(np.argmax(combined))
    best_lam = records[best_idx]["lam"]

    print(f"\nBest lambda = {best_lam:.6f} (combined = {combined[best_idx]:.4f}, "
          f"ve = {ve_vals[best_idx]:.4f}, pred_err = {pred_err_vals[best_idx]:.4f})")
    return best_lam


def compare_bair_vs_manifold(X_raw, y_raw, k=2, test_size=0.2, random_state=42, steps=200):
    import time
    from sklearn.model_selection import train_test_split
    from models.spca_bair import SPCA_Bair
    from models.spca_manifold import SupervisedPCA

    device = X_raw.device

    X_np = X_raw.cpu().numpy()
    y_np = y_raw.cpu().numpy()
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state
    )

    X_train = torch.tensor(X_train_np, dtype=torch.float64, device=device)
    X_test = torch.tensor(X_test_np, dtype=torch.float64, device=device)
    y_train = torch.tensor(y_train_np, dtype=torch.float64, device=device)
    y_test = torch.tensor(y_test_np, dtype=torch.float64, device=device)

    train_mean_X = X_train.mean(dim=0)
    train_mean_y = y_train.mean(dim=0)
    X_train_centered = X_train - train_mean_X
    X_test_centered = X_test - train_mean_X
    y_train_centered = y_train - train_mean_y
    y_test_centered = y_test - train_mean_y

    eps = 1e-8
    X_min = X_train_centered.min(dim=0, keepdim=True).values
    X_max = X_train_centered.max(dim=0, keepdim=True).values
    X_train = 2 * (X_train_centered - X_min) / (X_max - X_min + eps) - 1
    X_test = 2 * (X_test_centered - X_min) / (X_max - X_min + eps) - 1

    y_min = y_train_centered.min(dim=0, keepdim=True).values
    y_max = y_train_centered.max(dim=0, keepdim=True).values
    y_train = 2 * (y_train_centered - y_min) / (y_max - y_min + eps) - 1
    y_test = 2 * (y_test_centered - y_min) / (y_max - y_min + eps) - 1

    X_train_np = X_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    y_train_np = y_train.squeeze(-1).cpu().numpy()

    # Bair
    best_theta = select_theta_bair(X_train_np, y_train_np, random_state=random_state)
    t0 = time.time()
    bair_model = SPCA_Bair(theta=best_theta, n_components=k)
    bair_model.fit(X_train_np, y_train_np)
    t_bair = time.time() - t0

    L_bair = torch.zeros((X_train.shape[1], k), dtype=X_train.dtype, device=X_train.device)
    L_bair[bair_model.selected_idx, :] = torch.tensor(
        bair_model.w.T, dtype=X_train.dtype, device=X_train.device
    )

    ve_bair = variation_explained(X_test, L_bair).item()
    pe_bair = prediction_error(X_test, L_bair, y_test).item()

    # Manifold
    best_lam = select_lambda_regression(X_train, y_train, random_state=random_state, steps=steps)
    t0 = time.time()
    manifold_model = SupervisedPCA(input_dim=X_train.shape[1], output_dim=k)
    manifold_model.train(X_train, y_train, lam=best_lam, steps=steps)
    t_manifold = time.time() - t0
    L_manifold = manifold_model.L.detach()

    ve_manifold = variation_explained(X_test, L_manifold).item()
    pe_manifold = prediction_error(X_test, L_manifold, y_test).item()

    return {
        'bair': {
            'theta': best_theta,
            've': ve_bair,
            'pred_err': pe_bair,
            'runtime': t_bair,
            'n_selected_features': len(bair_model.selected_idx),
            'L': L_bair,
        },
        'manifold': {
            'lam': best_lam,
            've': ve_manifold,
            'pred_err': pe_manifold,
            'runtime': t_manifold,
            'L': L_manifold,
        },
    }


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
    zoo_feature_names = {
        "hair": "dlake",
        "feathers": "perje",
        "eggs": "jajca",
        "milk": "mleko",
        "airborne": "letenje",
        "aquatic": "vodne živali",
        "predator": "plenilec",
        "toothed": "zobje",
        "backbone": "hrbtenica",
        "breathes": "dihanje",
        "venomous": "strupen",
        "fins": "plavuti",
        "legs": "noge",
        "tail": "rep",
        "domestic": "udomačen",
        "catsize": "velikost",
    }
    feature_names = [zoo_feature_names[col] for col in feature_cols]
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
    zoo_class_names = {
        "amphibian": "dvoživke",
        "bird": "ptiči",
        "fish": "ribe",
        "insect": "žuželke",
        "invertebrate": "nevretenčarji",
        "mammal": "sesalci",
        "reptile": "plazilci",
    }
    class_names = [zoo_class_names[c] for c in types_cat.cat.categories]

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

def load_energy_efficiency_data(device=None):
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=242)
    df = data.data.features
    targets = data.data.targets
    X = torch.tensor(df.values, dtype=torch.float64, device=device)
    y = targets["Y1"].values  # Heating Load
    y = torch.tensor(y, dtype=torch.float64, device=device).unsqueeze(1)
    return X, y


def load_real_estate_data(device=None):
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=477)
    df = data.data.features
    target = data.data.targets
    # Drop transaction date (X1) as instructed
    df = df.drop(columns=["X1 transaction date"])
    X = torch.tensor(df.values, dtype=torch.float64, device=device)
    y = target.values.ravel()
    y = torch.tensor(y, dtype=torch.float64, device=device).unsqueeze(1)
    return X, y


def load_wine_quality_data(device=None, n_samples=1000, random_state=42):
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=186)
    df = data.data.features
    target = data.data.targets
    rng = np.random.default_rng(random_state)
    keep = rng.choice(len(df), n_samples, replace=False)
    df = df.iloc[keep]
    target = target.iloc[keep]
    X = torch.tensor(df.values, dtype=torch.float64, device=device)
    y = target.values.ravel()
    y = torch.tensor(y, dtype=torch.float64, device=device).unsqueeze(1)
    return X, y


def load_student_performance_data(device=None):
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=320)
    df = data.data.features
    targets = data.data.targets
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    df_encoded = df_encoded.astype(float)
    X = torch.tensor(df_encoded.values, dtype=torch.float64, device=device)
    y = targets["G3"].values  # final grade only
    y = torch.tensor(y, dtype=torch.float64, device=device).unsqueeze(1)
    return X, y


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

def load_mushroom_data(device=None):
    from ucimlrepo import fetch_ucirepo
    mushroom = fetch_ucirepo(id=73)
    df = mushroom.data.features
    target = mushroom.data.targets["poisonous"]

    # drop constant column
    df = df.drop(columns=["veil-type"])

    # select 1000 random samples
    rng = np.random.default_rng(0)
    keep = rng.choice(len(df), 1000, replace=False)
    df = df.iloc[keep]
    target = target.iloc[keep]

    feature_names = df.columns.tolist()
    df_encoded = pd.get_dummies(df, columns=feature_names)
    mushroom_feature_names = {
        "cap-shape": "oblika_klobuka",
        "cap-surface": "površina_klobuka",
        "cap-color": "barva_klobuka",
        "bruises": "modrice",
        "odor": "vonj",
        "gill-attachment": "pritrditev_lističev",
        "gill-spacing": "razmik_lističev",
        "gill-size": "velikost_lističev",
        "gill-color": "barva_lističev",
        "stalk-shape": "oblika_beta",
        "stalk-root": "podstavek_beta",
        "stalk-surface-above-ring": "površina_beta_nad_obročkom",
        "stalk-surface-below-ring": "površina_beta_pod_obročkom",
        "stalk-color-above-ring": "barva_beta_nad_obročkom",
        "stalk-color-below-ring": "barva_beta_pod_obročkom",
        "veil-type": "vrsta_koprene",
        "veil-color": "barva_koprene",
        "ring-number": "število_obročkov",
        "ring-type": "vrsta_obročka",
        "spore-print-color": "barva_trosov",
        "population": "populacija",
        "habitat": "življenjski_prostor",
    }
    encoded_feature_names = []
    for col in df_encoded.columns.tolist():
        base, val = col.rsplit("_", 1)
        encoded_feature_names.append(f"{mushroom_feature_names.get(base, base)}_{val}")

    E_raw = torch.tensor(
        df_encoded.values,
        dtype=torch.float32,
        device=device
    )

    class_cat = target.astype("category")
    C = torch.tensor(
        class_cat.cat.codes.values,
        dtype=torch.long,
        device=device
    )
    mushroom_class_names = {'e': 'užitna', 'p': 'strupena'}
    class_names = [mushroom_class_names[c] for c in class_cat.cat.categories]

    instance_names = [str(i) for i in range(len(df))]

    return E_raw, C, instance_names, encoded_feature_names, class_names

def load_glass_data(device=None):
    from ucimlrepo import fetch_ucirepo
    glass = fetch_ucirepo(id=42)
    df = glass.data.features
    target = glass.data.targets["Type_of_glass"]

    feature_names = df.columns.tolist()
    E_raw = torch.tensor(
        df.values,
        dtype=torch.float32,
        device=device
    )

    glass_type_names = {
        1: "building_windows_float",
        2: "building_windows_non_float",
        3: "vehicle_windows_float",
        4: "vehicle_windows_non_float",
        5: "containers",
        6: "tableware",
        7: "headlamps",
    }
    class_cat = target.astype("category")
    C = torch.tensor(
        class_cat.cat.codes.values,
        dtype=torch.long,
        device=device
    )
    class_names = [glass_type_names[int(c)] for c in class_cat.cat.categories]
    instance_names = [str(i) for i in range(len(df))]

    return E_raw, C, instance_names, feature_names, class_names

def load_breast_cancer_data(device=None):
    from ucimlrepo import fetch_ucirepo
    bc = fetch_ucirepo(id=17)
    df = bc.data.features
    target = bc.data.targets["Diagnosis"]

    # 30 features are 10 base measurements x 3 statistics (mean, se, worst).
    # Keep only the mean values.
    mean_cols = [c for c in df.columns if c.endswith("1")]
    bc_feature_names = {
        "radius": "polmer",
        "texture": "tekstura",
        "perimeter": "obseg",
        "area": "površina",
        "smoothness": "gladkost",
        "compactness": "kompaktnost",
        "concavity": "konkavnost",
        "concave_points": "konkavne točke",
        "symmetry": "simetrija",
        "fractal_dimension": "fraktalna dimenzija",
    }
    feature_names = [bc_feature_names[c.replace("1", "")] for c in mean_cols]
    E_raw = torch.tensor(
        df[mean_cols].values,
        dtype=torch.float32,
        device=device
    )

    class_cat = target.astype("category")
    C = torch.tensor(
        class_cat.cat.codes.values,
        dtype=torch.long,
        device=device
    )
    bc_class_names = {'B': 'neškodljiv', 'M': 'rakotvoren'}
    class_names = [bc_class_names[c] for c in class_cat.cat.categories]
    instance_names = [str(i) for i in range(len(df))]

    return E_raw, C, instance_names, feature_names, class_names

def plot_projection(points, vectors, feature_names, class_codes, class_names, title, loading_cutoff=0.4, ax=None, text_scale=1.1, top_n_loadings=None, normalize_points=False, max_points=None, title_fontsize=None):
    n = points.shape[0]

    if normalize_points:
        max_dist = np.max(np.sqrt(points[:, 0]**2 + points[:, 1]**2))
        if max_dist > 0:
            points = points / max_dist

    if max_points is None:
        max_points = 100

    if n > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(n, max_points, replace=False)
        points = points[keep]
        class_codes = class_codes[keep]

    if top_n_loadings is not None:
        lengths = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        sorted_lengths = np.sort(lengths)[::-1]
        cutoff = sorted_lengths[min(top_n_loadings - 1, len(sorted_lengths) - 1)]
    else:
        cutoff = loading_cutoff

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

    circle = plt.Circle((0, 0), cutoff, fill=False, linestyle='--', linewidth=0.8, color='gray')
    ax.add_patch(circle)

    above = [(i, vec) for i, vec in enumerate(vectors) if np.sqrt(vec[0]**2 + vec[1]**2) > cutoff]

    texts = []
    for i, (x, y) in above:
        ax.plot([0, x], [0, y], linestyle='-', linewidth=1.2, color='black')
        t = ax.text(
            x * text_scale, y * text_scale,
            feature_names[i],
            fontsize=8,
            ha='center',
            va='center'
        )
        texts.append(t)

    from adjustText import adjust_text
    from matplotlib.collections import PathCollection
    scatter_objs = [c for c in ax.collections if isinstance(c, PathCollection)]
    adjust_text(
        texts,
        objects=scatter_objs,
        ax=ax,
        autoalign='xy',
        expand=(1.2, 1.5),
        force_text=(0.5, 1.0),
        force_points=(0.5, 0.5),
        arrowprops=dict(arrowstyle='-', color='grey', lw=0.5, alpha=0.5),
        ensure_no_overlap=True,
    )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(title='Razredi', loc='upper right', frameon=True, fontsize=9, title_fontsize=10)
    ax.set_title(title, fontsize=title_fontsize)

    return ax