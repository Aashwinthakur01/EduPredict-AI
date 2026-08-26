from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ==========================================
# LOAD TRAINED ML MODEL
# ==========================================

model = joblib.load("model.pkl")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # ==========================================
    # 1. GET INPUTS FROM FORM
    # ==========================================

    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    previous_marks = float(request.form["previous_marks"])
    assignment_completion = float(
        request.form["assignment_completion"]
    )
    sleep_hours = float(request.form["sleep_hours"])
    participation = float(request.form["participation"])
    backlogs = float(request.form["backlogs"])


    # ==========================================
    # 2. PREPARE INPUT FOR MACHINE LEARNING MODEL
    # ==========================================

    features = pd.DataFrame(
        [[
            attendance,
            study_hours,
            previous_marks,
            assignment_completion,
            sleep_hours,
            participation,
            backlogs
        ]],
        columns=[
            "attendance",
            "study_hours",
            "previous_marks",
            "assignment_completion",
            "sleep_hours",
            "participation",
            "backlogs"
        ]
    )


    # ==========================================
    # 3. PREDICT PERFORMANCE
    # ==========================================

    prediction = model.predict(features)[0]

    # Keep prediction between 0 and 100
    prediction = max(0, min(100, prediction))

    prediction = round(prediction, 2)


    # ==========================================
    # 4. DETERMINE PERFORMANCE LEVEL
    # ==========================================

    if prediction >= 85:

        performance_level = "Excellent"

        recommendation = (
            "Your academic performance is excellent. "
            "Maintain your current study routine and focus "
            "on advanced concepts, practical projects and "
            "skill development."
        )

    elif prediction >= 70:

        performance_level = "Good"

        recommendation = (
            "Your academic performance is good. "
            "With a little more consistency in study, "
            "revision and academic activities, you can "
            "move toward an excellent performance level."
        )

    elif prediction >= 50:

        performance_level = "Average"

        recommendation = (
            "Your performance is average and has clear "
            "scope for improvement. Focus on consistent "
            "study, revision, attendance and assignment completion."
        )

    else:

        performance_level = "Needs Improvement"

        recommendation = (
            "Your current performance needs significant attention. "
            "Follow a structured study schedule, improve attendance, "
            "complete assignments and focus on clearing backlogs."
        )


    # ==========================================
    # 5. PERSONALIZED RECOMMENDATION ENGINE
    # ==========================================

    tips = []


    # ---------- ATTENDANCE ----------

    if attendance < 60:

        tips.append(
            "🔴 Attendance is critically low. "
            "Try to attend classes regularly and work toward "
            "at least 75% attendance."
        )

    elif attendance < 75:

        tips.append(
            "🟠 Your attendance is below 75%. "
            "Improve class attendance to maintain academic consistency."
        )

    else:

        tips.append(
            "🟢 Your attendance is at a healthy level. "
            "Continue maintaining regular attendance."
        )


    # ---------- STUDY HOURS ----------

    if study_hours < 2:

        tips.append(
            "🔴 Your daily study time is very low. "
            "Gradually increase focused study time to around 3–4 hours."
        )

    elif study_hours < 3:

        tips.append(
            "🟠 Increase your daily focused study time. "
            "Try adding at least 1 extra hour of productive study."
        )

    else:

        tips.append(
            "🟢 Your daily study time is good. "
            "Focus on maintaining quality and consistency."
        )


    # ---------- PREVIOUS MARKS ----------

    if previous_marks < 50:

        tips.append(
            "🔴 Your previous marks indicate that fundamental concepts "
            "may need more attention. Start with basic concepts and "
            "practice questions regularly."
        )

    elif previous_marks < 60:

        tips.append(
            "🟠 Revise important concepts from previous topics "
            "and practice previous-year or topic-wise questions."
        )

    else:

        tips.append(
            "🟢 Your previous academic performance is relatively strong. "
            "Use it as a foundation for further improvement."
        )


    # ---------- ASSIGNMENTS ----------

    if assignment_completion < 50:

        tips.append(
            "🔴 Assignment completion is low. "
            "Create a weekly deadline plan and complete pending work first."
        )

    elif assignment_completion < 70:

        tips.append(
            "🟠 Try to improve assignment completion above 70% "
            "by following a regular submission schedule."
        )

    else:

        tips.append(
            "🟢 Your assignment completion is good. "
            "Continue submitting work on time."
        )


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
            "🟢 Your sleep duration is within a balanced range. "
            "Continue maintaining a consistent sleep routine."
        )


    # ---------- CLASS PARTICIPATION ----------

    if participation < 40:

        tips.append(
            "🔴 Class participation is very low. "
            "Ask questions, participate in discussions and engage "
            "more actively during lectures."
        )

    elif participation < 60:

        tips.append(
            "🟠 Try to participate more actively in class discussions "
            "and activities."
        )

    else:

        tips.append(
            "🟢 Your class participation is good. "
            "Continue engaging actively in learning activities."
        )


    # ---------- BACKLOGS ----------

    if backlogs >= 3:

        tips.append(
            "🔴 You have multiple backlogs. "
            "Create a priority-based plan and focus on clearing "
            "pending subjects one by one."
        )

    elif backlogs > 0:

        tips.append(
            "🟠 You have pending subjects. "
            "Allocate dedicated weekly time to clear your backlogs."
        )

    else:

        tips.append(
            "🟢 You currently have no backlogs. "
            "Maintain this consistency."
        )


    # ==========================================
    # 6. SEND DATA TO RESULT PAGE
    # ==========================================

    return render_template(
        "result.html",
        prediction=prediction,
        performance_level=performance_level,
        recommendation=recommendation,
        tips=tips
    )


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)