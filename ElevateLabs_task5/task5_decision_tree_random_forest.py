# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import graphviz

# 2. Load dataset
data = pd.read_csv("heart.csv")  # Ensure heart.csv is in the same directory
print("First 5 rows of the dataset:")
print(data.head())

# 3. Check for missing values
print("\nMissing values in dataset:")
print(data.isnull().sum())

# 4. Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# 5. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n🎯 Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# 7. Visualize Decision Tree
dot_data = export_graphviz(dt, out_file=None,
                           feature_names=X.columns,
                           class_names=['No Disease', 'Disease'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saves as decision_tree.pdf
graph.view()

# 8. Train Pruned Decision Tree (to reduce overfitting)
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_pruned = dt_pruned.predict(X_test)
print("\n🌳 Pruned Decision Tree Accuracy:", accuracy_score(y_test, y_pred_pruned))

# 9. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# 10. Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 11. Evaluate using Cross-Validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print("\n📊 Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
