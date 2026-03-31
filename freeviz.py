import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from math import sqrt, sin, cos, pi
import pandas as pd

def normalize(A):
    """
    Longest base vector projection = 1
    Sum of all base vector projections = 0 
    """
    A = A - A.mean(axis=0, keepdims=True)
    return A / np.max(np.linalg.norm(A, axis=1))

# one iteration
def step(E, C, A):
    P = E @ A / E.sum(axis=1, keepdims=True)
    F = np.zeros(P.shape)
    G = np.zeros(A.shape)
    n_instances = E.shape[0]
    n_features = A.shape[0]

    for e in range(n_instances):
        for f in range(e+1, n_instances):
            dx = P[e][0] - P[f][0]
            dy = P[e][1] - P[f][1]
            r = sqrt(dx**2 + dy**2)
            if r == 0:
                r = 0.001

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

    for e in range(n_instances):
        for i in range(n_features):
            G[i][0] += F[e][0] * E[e][i]
            G[i][1] += F[e][1] * E[e][i]

    lr = 0.01
    A = A + lr*G
    A = normalize(A)
    Wp = np.sum(np.abs(F))
    return A, Wp

def vizualize(P, C, A, feature_names, class_names, names):

    # scatter instance projections
    plt.scatter(P[:, :1], P[:, 1:], c=C)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.gca().set_aspect('equal', 'box')

    # Plot each row of A as a vector from the origin
    for i, vec in enumerate(A):
        x, y = vec  # (A_i0, A_i1)
        if sqrt(x**2 + y**2) > 0.5:
            plt.plot([0, x], [0, y], linestyle='-', linewidth=2) 
            plt.text(x * 1.05, y * 1.05, feature_names[i], ha='left', va='bottom')

    # Enable hover tooltips
    cursor = mplcursors.cursor(hover=True) 
    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        cls = C[i]
        cls_name = class_names[cls] 
        sel.annotation.set_text(f"{names[i]}\n{cls_name}")
    plt.show()

def initialize_projection_matrix(n_dim):
    step = 2*pi/n_dim
    A = np.zeros([n_dim, 2])
    for i in range(n_dim):
        A[i] = [sin((i*step)), cos((i*step))]
    return A

def load_zoo_tab(path):
    df = pd.read_csv(path, sep="\t", skiprows=[1, 2])
    instance_names = df["name"].tolist()
    feature_cols = [col for col in df.columns if col not in ("name", "type")]
    feature_names = feature_cols
    E_raw = df[feature_cols].values.astype(float)
    types_cat = df["type"].astype("category") # C
    C = types_cat.cat.codes.values          # 0,1,2,... for each class
    class_names = list(types_cat.cat.categories)
    return E_raw, C, instance_names, feature_names, class_names

def scale_features(E_raw):
    E_min = E_raw.min(axis=0, keepdims=True)
    E_max = E_raw.max(axis=0, keepdims=True)
    E = (E_raw - E_min) / np.where(E_max - E_min == 0, 1, E_max - E_min)
    return E

if __name__ == "__main__":
    # Initialize matrices
    E_raw, C, names, feature_names, class_names = load_zoo_tab("data/zoo.tab")
    E = scale_features(E_raw)
    A = initialize_projection_matrix(E.shape[1])

    # Run a step
    A, Wp_old = step(E, C, A)
    i = 1
    while(1):
        i += 1
        A, Wp = step(E, C, A)
        print(Wp)
        if abs(Wp_old - Wp) / Wp_old < 0.001:
            break
        Wp_old = Wp
    print("steps: ", i)
    P = E @ A / E.sum(axis=1, keepdims=True)

    vizualize(P, C, A, feature_names, class_names, names)
