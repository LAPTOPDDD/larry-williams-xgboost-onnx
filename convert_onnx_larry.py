import joblib
import json
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from xgboost import XGBClassifier

# 1. REGISTER CONVERTER
update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'zipmap': [True, False], 'nocl': [True, False]} 
)

def prepare_model_for_onnx(model):
    """Fix base_score and reset feature names to prevent ONNX conversion errors"""
    try:
        booster = model.get_booster()
        
        # --- FIX BASE SCORE ---
        config = json.loads(booster.save_config())
        bs_raw = config['learner']['learner_model_param']['base_score']
        if isinstance(bs_raw, str):
            clean_str = bs_raw.replace('[', '').replace(']', '').split(',')[0]
            model.base_score = float(clean_str)
        
        # --- FIX FEATURE NAMES (KEY FIX FOR YOUR ERROR) ---
        # Force XGBoost to use generic names (f0, f1, etc) instead of custom names
        num_features = model.n_features_in_
        new_names = [f'f{i}' for i in range(num_features)]
        booster.feature_names = new_names
        model._Booster = booster
        
        print(f"Model prepared: Base_score fixed & Features renamed to f0-f{num_features-1}")
    except Exception as e:
        print(f"Warning during preparation: {e}")

# 2. LOAD AND PREPARE MODEL
model = joblib.load('models/larry_model.pkl')
prepare_model_for_onnx(model)

# 3. CONVERT TO ONNX
initial_type = [('float_input', FloatTensorType([None, 10]))]  # 10 features as per the training
options = {'zipmap': False} 

print("Converting to ONNX...")
try:
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_type, 
        options=options,
        target_opset={'': 12, 'ai.onnx.ml': 3}
    )

    # 4. SAVE ONNX MODEL
    with open("models/larry_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Success! File larry_model.onnx is ready for use in MQL5.")

except Exception as e:
    print(f"Error during conversion: {e}")