from flask import Flask, request, jsonify

# Inisialisasi Flask
app = Flask(__name__)

# Memuat model joblib
model = joblib.load('ridge_student_exam_prediction.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil JSON dengan format {"data": {fitur: nilai}}
        json_data = request.json
        # Ubah menjadi DataFrame
        input_df = pd.DataFrame([json_data['data']])
        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
