import joblib

def predict(data):
    model = joblib.load('ml_app/ml/model.pkl')
    return model.predict(data)