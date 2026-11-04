import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Input and output
X = data[["study_hours"]]
y = data["marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict marks
predictions = model.predict(X_test)

# Evaluate model
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Predict for a new student
hours = float(input("Enter study hours: "))

predicted_marks = model.predict(
    pd.DataFrame({"study_hours": [hours]})
)

print("Predicted Marks:", predicted_marks[0])

# Visualize results
plt.scatter(X, y, label="Actual Marks")
plt.plot(X, model.predict(X), color="red", label="Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()
