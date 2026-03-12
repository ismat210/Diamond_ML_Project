# test_pred.py
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.predict_clarity_pipeline import CustomClarityData, PredictClarityPipeline

# ==============================
# Input your diamond features here
# ==============================
diamond_input = {
    "carat": 1.2,
    "depth": 61.5,
    "table": 55.0,
    "x": 6.8,
    "y": 6.9,
    "z": 4.2,
    "cut": "Premium",
    "color": "G",
    "clarity": "VS2"  # For price model input only
}

# ==============================
# Price Prediction
# ==============================
price_data = CustomData(**diamond_input)
price_df = price_data.get_data_as_dataframe()

price_model = PredictPipeline()
price_pred = price_model.predict(price_df)
price_pred = round(price_pred[0], 2)
print(f"Predicted Price: ${price_pred}")

# ==============================
# Clarity Prediction
# ==============================
clarity_data = CustomClarityData(
    carat=diamond_input["carat"],
    depth=diamond_input["depth"],
    table=diamond_input["table"],
    x=diamond_input["x"],
    y=diamond_input["y"],
    z=diamond_input["z"],
    cut=diamond_input["cut"],
    color=diamond_input["color"]
)

clarity_df = clarity_data.get_data_as_dataframe()
clarity_model = PredictClarityPipeline()
clarity_pred = clarity_model.predict(clarity_df)

# Map numeric prediction back to clarity grade
clarity_mapping = {
    1: "I1",
    2: "SI2",
    3: "SI1",
    4: "VS2",
    5: "VS1",
    6: "VVS2",
    7: "VVS1",
    8: "IF"
}

clarity_label = clarity_mapping[int(clarity_pred[0])]
print(f"Predicted Clarity: {clarity_label}")