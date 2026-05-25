"""Quick train script to produce model artifacts when dataset is unavailable."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clique.algorithm import CLIQUE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from clique.utils import FEATURE_NAMES, save_model

data_path = ROOT / "data" / "test_clean.csv"
profiles = pd.read_csv(data_path)
X = profiles[FEATURE_NAMES].values.astype(float)
X_train, _ = train_test_split(X, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
scaler.fit(X_train)
(ROOT / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, ROOT / "models" / "scaler.pkl")
pd.DataFrame(X_train, columns=FEATURE_NAMES).to_csv(ROOT / "models" / "X_train_scaled.csv", index=False)
model = CLIQUE(xi=8, tau=0.05)
model.fit(X_train, feature_names=FEATURE_NAMES)
save_model(model, scaler, model.cluster_profiles_, str(ROOT / "models"))
print(f"Clusters: {len(model.clusters_)}, labels unique: {np.unique(model.labels_)}")
