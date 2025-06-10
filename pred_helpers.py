import joblib
import numpy as np

#model=joblib.load("models/rf_model.pkl")
model=joblib.load("models/logistic_regression_model.pkl")
#model=joblib.load("models/mlp_model.pth")
feature_col = ["Slice_eMBB","Slice_URLLC","Slice_mMTC","Asked","Admitted","Dropped"]

#try:
#    SCALER = joblib.load("num_scaler.pkl")
#except FileNotFoundError:
#    SCALER = None

def _build_row(slice_name: str, ask: float, adm: float, drp: float) -> np.ndarray:

    row = {c: 0.0 for c in feature_col}
    one_hot_key = f"Slice_{slice_name}"
    if one_hot_key in row:
        row[one_hot_key]=1.0

    row["Asked"]=ask
    row["Admitted"]=adm
    row["Dropped"]=drp

    X = np.array([[row[c] for c in feature_col]],dtype=float)


    return X

def predict_violation(slice_name: str, ask: float, adm: float, drp: float) -> int:
    X=_build_row(slice_name,ask,adm,drp)
    
    return int(model.predict(X)[0])
