from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    attendance = float(request.form.get('attendance', 85))
    study_hours = float(request.form.get('study_hours', 5))
    previous_marks = float(request.form.get('previous_marks', 75))
    assignment_completion = float(request.form.get('assignment_completion', 80))
    sleep_hours = float(request.form.get('sleep_hours', 7))
    participation = float(request.form.get('participation', 6))
    backlogs = float(request.form.get('backlogs', 0))

    features = [[attendance, study_hours, previous_marks, assignment_completion, sleep_hours, participation, backlogs]]
    prediction = model.predict(features)[0]
    prediction = round(float(prediction), 2)

    if prediction >= 85:
        category = "Excellent Performance"
        message = "Outstanding! Keep it up."
    elif prediction >= 70:
        category = "Good Performance"
        message = "You're doing well! Small improvements can take you to excellent."
    elif prediction >= 50:
        category = "Average Performance"
        message = "You have potential. Focus on consistency."
    else:
        category = "Needs Improvement"
        message = "Don't worry, you can improve with right strategy."

    tips = []
    if sleep_hours < 6:
        tips.append("😴 Your sleep duration is low. Maintain a healthier sleep schedule.")
    elif sleep_hours > 9:
        tips.append("😴 Sleep is high. Balance study and sleep.")
    else:
        tips.append("😴 Sleep is balanced. Continue consistent routine.")

    if study_hours < 3:
        tips.append("📚 Increase study hours to at least 4-5 hours.")
    if attendance < 75:
        tips.append("🎯 Attendance is low. Try to maintain above 85%.")

    return render_template('result.html', prediction=prediction, category=category, message=message, tips=tips)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)