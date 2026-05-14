import joblib
from config import BEST_MODEL_PATH

def __init__(self):
    self.model_path = BEST_MODEL_PATH

    if not self.model_path.exists():
        raise FileNotFoundError(
        "Model not found. Please train the model first."
        
        )

    self.model = joblib.load(self.model_path)