from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved ML model (relative path)
# model_path = "svm_depression_detection_model.pkl"
model_path="mlp_model.pkl"
svm_model = joblib.load(model_path)


# Mapping responses to numerical values
response_mapping = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

# Depression message function
def get_depression_message(level):
    if level == "High":
        return "You have high depressive disorder. Visit a doctor for counselling."
    elif level == "Severe":
        return "You have severe depressive disorder. You need immediate medical attention and critical care."
    elif level == "Moderate":
        return "You have moderate depression. Consider consulting a doctor."
    elif level == "Mild":
        return "You have mild depression. Be watchful!"
    else:
        return "Your mental health seems stable. Stay positive and take care!"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form inputs
        user_input = [
            request.form.get(f"q{i+1}") for i in range(9)
        ]
        
        # Convert to numerical values
        data = pd.DataFrame([[response_mapping[val] for val in user_input]],
                            columns=['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9'])
        
        # Make prediction
        prediction = svm_model.predict(data)[0]  # Get single prediction
        
        # Get message
        message = get_depression_message(prediction)
        
        return render_template("index.html", prediction=prediction, message=message)
    
    return render_template("index.html", prediction=None, message=None)

if __name__ == "__main__":
    app.run(debug=True)
