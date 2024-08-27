import pandas as pd
import time
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

# Start timing the program execution
start_time = time.time()

# Load the dataset
file_path = r'D:\srinidhi\amrita\MFC\SMOTE_analysis.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X = df.iloc[:, :-1]  # Features (embed_0 to embed_383)
y = df['Classification']  # Target

# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the parameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X, y)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)

# Now, you can use the best parameters found by grid search for Random Forest
best_rf = RandomForestClassifier(**best_params, random_state=42)

# Perform cross-validation and calculate evaluation metrics
accuracy = cross_val_score(best_rf, X, y, cv=kf, scoring='accuracy')
precision = cross_val_score(best_rf, X, y, cv=kf, scoring='precision_macro')
recall = cross_val_score(best_rf, X, y, cv=kf, scoring='recall_macro')
f1 = cross_val_score(best_rf, X, y, cv=kf, scoring='f1_macro')

# Print the results
print("\nPrecision:")
print("  - Mean: {:.2f}".format(precision.mean()))
print("  - Standard Deviation: {:.2f}".format(precision.std()))

print("\nRecall:")
print("  - Mean: {:.2f}".format(recall.mean()))
print("  - Standard Deviation: {:.2f}".format(recall.std()))

print("\nF1 Score:")
print("  - Mean: {:.2f}".format(f1.mean()))
print("  - Standard Deviation: {:.2f}".format(f1.std()))

print("\nAccuracy:")
print("  - Mean: {:.2f}".format(accuracy.mean()))
print("  - Standard Deviation: {:.2f}".format(accuracy.std()))

# Load testing data
testing_data = pd.read_excel(r'D:\srinidhi\amrita\MFC\sample_test.xlsx')

# Assuming you already have 'X_test' containing the features (embeddings) for testing
X_test = testing_data  # No need to exclude the last column

best_rf.fit(X, y)

# Predict classifications for testing data
predictions = best_rf.predict(X_test)

# Add predictions to the testing data DataFrame
testing_data['Predicted_Classification'] = predictions

# Save the testing data with predicted classifications to a new Excel file
testing_output_file = "predicted_testing_t5_rf.xlsx"
testing_data.to_excel(testing_output_file, index=False)
print("Predictions saved to:", testing_output_file)

# End timing the program execution
end_time = time.time()
execution_time = end_time - start_time

print("Time taken for running the whole program: {:.2f} seconds".format(execution_time))