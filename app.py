from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    study_hours = float(request.form.get('study_hours', 5))
    sleep_hours = float(request.form.get('sleep_hours', 7))
    attendance = float(request.form.get('attendance', 85))
    previous_score = float(request.form.get('previous_score', 75))
    class_participation = float(request.form.get('class_participation', 6))
    extracurricular = float(request.form.get('extracurricular', 2))
    motivation = float(request.form.get('motivation', 7))
    stress = float(request.form.get('stress', 4))

    # Prediction (your model)
    features = [[study_hours, sleep_hours, attendance, previous_score, class_participation, extracurricular, motivation, stress]]
    prediction = model.predict(features)[0]
    prediction = round(prediction, 2)

    # Category logic
    if prediction >= 85:
        category = "Excellent Performance"
        message = "Outstanding! Keep up the great work."
    elif prediction >= 70:
        category = "Good Performance"
        message = "You're doing well! Small improvements can take you to excellent."
    elif prediction >= 50:
        category = "Average Performance"
        message = "You have potential. Focus on consistency."
    else:
        category = "Needs Improvement"
        message = "Don't worry, with right strategy you can improve."

    # -------- SLEEP LOGIC FROM YOUR SCREENSHOT --------
    tips = []

    # ---------- SLEEP ----------
    if sleep_hours < 6:
        tips.append(
            "😴 Your sleep duration is low. "
            "Maintain a healthier sleep schedule to support concentration "
            "and learning."
        )
    elif sleep_hours > 9:
        tips.append(
            "😴 Your sleep duration is relatively high. "
            "Maintain a balanced routine while ensuring enough time "
            "for study and other activities."
        )
    else:
        tips.append(
            "😴 Your sleep duration is within a balanced range. "
            "Continue maintaining a consistent sleep routine."
        )

    # Add more tips for other factors
    if study_hours < 3:
        tips.append("📚 Increase study hours to at least 4-5 hours for better retention.")
    if attendance < 75:
        tips.append("🎯 Attendance is low. Try to maintain above 85% for better performance.")

    return render_template('result.html',
                           prediction=prediction,
                           category=category,
                           message=message,
                           tips=tips)

if __name__ == '__main__':
    app.run(debug=True)