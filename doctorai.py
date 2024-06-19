import numpy as np
import pandas as pd
from google.colab import files
uploaded=files.upload()
data=pd.read_csv('DoctorAI Final.csv')
data
from sklearn.preprocessing import LabelEncoder
cat_cols = ['Disease','Age Group','Medicine Name','Male Dosage','Female Dosage','Side Effects','Max Tablets/Day','Foods to eat']
label_encoders = {}
for col in cat_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

x = data[['Disease', 'Age Group']]
y = data['Medicine Name']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
x
y
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)
best_rf = grid_search.best_estimator_
best_rf.fit(x_train, y_train)
y_pred = best_rf.predict(x_test)
train_accuracy = best_rf.score(x_train, y_train)
test_accuracy = best_rf.score(x_test, y_test)

print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.suptitle('MED PRED', color='r', size=30)
plt.show()




