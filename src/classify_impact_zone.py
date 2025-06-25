import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# ✅ Step 1: Load the CSV and assign column names
df = pd.read_csv("impact_points.csv", header=None, names=["Image", "X", "Y", "Type"])

# ✅ Step 2: Rule-based classifier function
def rule_based_classifier(x, y):
    if 100 <= x <= 160 and 400 <= y <= 480:
        return "Sweet Spot"
    elif x < 100 and y > 480:
        return "Toe"
    elif x > 160 and y > 480:
        return "Edge"
    elif y < 300:
        return "Upper Edge"
    elif x < 100 and y < 300:
        return "Handle"
    else:
        return "Missed"

# ✅ Step 3: Apply the rule-based classifier
df['Predicted_Rule'] = df.apply(lambda row: rule_based_classifier(row['X'], row['Y']), axis=1)

print("\n🧠 Rule-Based Classifier Output:")
print(df[['X', 'Y', 'Type', 'Predicted_Rule']])

# ✅ Step 4: Train a Decision Tree classifier
X = df[['X', 'Y']]
y = df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# ✅ Step 5: Evaluate the ML model
print("\n📊 Machine Learning Model Evaluation:")
print(classification_report(y_test, y_pred))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
