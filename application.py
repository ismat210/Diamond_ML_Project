# application.py
from doctest import debug

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin

# Import your pipelines
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.predict_clarity_pipeline import CustomClarityData, PredictClarityPipeline

# ===========================
# Flask App Setup
# ===========================
application = Flask(__name__)
app = application
CORS(app)

# ===========================
# Home Page
# ===========================
@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')


# ===========================
# Form Prediction (GET/POST)
# ===========================
@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            # -------- Price Inputs --------
            price_data = CustomData(
                carat=float(request.form.get('carat')),
                depth=float(request.form.get('depth')),
                table=float(request.form.get('table')),
                x=float(request.form.get('x')),
                y=float(request.form.get('y')),
                z=float(request.form.get('z')),
                cut=request.form.get('cut'),
                color=request.form.get('color'),
                clarity=request.form.get('clarity')
            )
            price_df = price_data.get_data_as_dataframe()

            # -------- Clarity Inputs --------
            clarity_data = CustomClarityData(
                carat=float(request.form.get('carat')),
                depth=float(request.form.get('depth')),
                table=float(request.form.get('table')),
                x=float(request.form.get('x')),
                y=float(request.form.get('y')),
                z=float(request.form.get('z')),
                cut=request.form.get('cut'),
                color=request.form.get('color')
            )
            clarity_df = clarity_data.get_data_as_dataframe()

            # -------- Run Pipelines --------
            price_pred_pipeline = PredictPipeline()
            price_pred = price_pred_pipeline.predict(price_df)
            price_result = round(price_pred[0], 2)

            clarity_pred_pipeline = PredictClarityPipeline()
            clarity_pred = clarity_pred_pipeline.predict(clarity_df)
            clarity_result = clarity_pred[0]  # predicted label

            # -------- Return to Template --------
            return render_template(
                'index.html',
                price_result=price_result,
                clarity_result=clarity_result,
                price_df=price_df,
                clarity_df=clarity_df
            )

        except Exception as e:
            return f"Error in prediction: {e}"


# ===========================
# API Endpoint
# ===========================
@app.route('/predictAPI', methods=['POST'])
@cross_origin()
def predict_api():
    try:
        data_json = request.json

        # -------- Price Prediction --------
        price_data = CustomData(
            carat=float(data_json['carat']),
            depth=float(data_json['depth']),
            table=float(data_json['table']),
            x=float(data_json['x']),
            y=float(data_json['y']),
            z=float(data_json['z']),
            cut=data_json['cut'],
            color=data_json['color'],
            clarity=data_json['clarity']
        )
        price_df = price_data.get_data_as_dataframe()
        price_pred_pipeline = PredictPipeline()
        price_pred = price_pred_pipeline.predict(price_df)

        # -------- Clarity Prediction --------
        clarity_data = CustomClarityData(
            carat=float(data_json['carat']),
            depth=float(data_json['depth']),
            table=float(data_json['table']),
            x=float(data_json['x']),
            y=float(data_json['y']),
            z=float(data_json['z']),
            cut=data_json['cut'],
            color=data_json['color']
        )
        clarity_df = clarity_data.get_data_as_dataframe()
        clarity_pred_pipeline = PredictClarityPipeline()
        clarity_pred = clarity_pred_pipeline.predict(clarity_df)

        # -------- Return JSON --------
        return jsonify({
            'price': round(price_pred[0], 2),
            'clarity': clarity_pred[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)})


# ===========================
# Run Server
# ===========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug = True)