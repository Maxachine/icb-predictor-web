# app.py

from flask import Flask, render_template, request, jsonify
import interface
import pdb

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [float(data[f'feature{i+1}']) for i in range(6)]
    indicators = interface.get_indicators_from_web(features)
    # pdb.set_trace()
    result = {'prediction':interface.compute_P(indicators)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)