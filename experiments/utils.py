def scale_features(E):
    E_min = E.min(dim=0, keepdim=True).values
    E_max = E.max(dim=0, keepdim=True).values
    E_scaled = (E - E_min) / (E_max - E_min)
    return E_scaled