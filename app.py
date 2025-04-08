# from flask import Flask, render_template, request, url_for
# import joblib
# from utils import clean_text

# # Initialize app
# app = Flask(__name__)

# # Load trained pipeline (includes TF-IDF and Logistic Regression)
# pipeline = joblib.load(r"D:\Chinmay\ML PROJECTS\Fake News Detection\AI_project\fake_news_model.pkl")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         text = request.form['news_text']
#         cleaned_text = clean_text(text)

#         # Direct prediction using the pipeline
#         prediction = pipeline.predict([cleaned_text])[0]
#         result = 'Fake' if prediction == 1 else 'Real'

#         return render_template(
#             'result.html',
#             prediction=result,
#             confusion_matrix=url_for('static', filename='confusion_matrix.png'),
#             roc_curve=url_for('static', filename='roc_curve.png'),
#             metrics=url_for('static', filename='metrics.png')
#         )

# if __name__ == '__main__':
#     app.run(debug=True)



# app.py
from flask import Flask, render_template, request, url_for
import joblib
from utils import clean_text

app = Flask(__name__)

# Load model and vectorizer
pipeline = joblib.load("fake_news_model.pkl")
model = pipeline['model']
vectorizer = pipeline['vectorizer']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['news_text']
        cleaned_text = clean_text(text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        result = 'Fake' if prediction == 1 else 'Real'

        return render_template(
            'result.html',
            prediction=result,
            confusion_matrix=url_for('static', filename='Confusion_Matrix.png'),
            roc_curve=url_for('static', filename='ROC_Curve.png'),
            metrics=url_for('static', filename='Classification_Report.png')
        )

if __name__ == '__main__':
    app.run(debug=True)
