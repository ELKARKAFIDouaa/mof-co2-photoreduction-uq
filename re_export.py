import joblib

obj = joblib.load("models/saved_models/gpr_matern_taux_log.pkl")
print("OK")
print(type(obj))