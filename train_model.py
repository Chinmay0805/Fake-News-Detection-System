# train_model.py
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report
import pandas as pd
import os
import seaborn as sns

# Load data
df = pd.read_csv("Shuffled_SAMPLE.csv")

# Preprocessing
X = df['Headline']
y = df['Label']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pipeline = {'model': model, 'vectorizer': vectorizer}
joblib.dump(pipeline, "fake_news_model.pkl")

# Create static/ folder if not exists
os.makedirs("static", exist_ok=True)

# Save confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.savefig("static/Confusion_Matrix.png")
plt.clf()

# Save ROC curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.savefig("static/ROC_Curve.png")
plt.clf()

# Save classification report as heatmap
report = classification_report(y_test, model.predict(X_test), output_dict=True)
df_report = pd.DataFrame(report).iloc[:-1, :].T  # Remove 'accuracy'

sns.heatmap(df_report, annot=True, cmap="Blues", fmt=".2f")
plt.title("Classification Report")
plt.savefig("static/Classification_Report.png")
plt.clf()
