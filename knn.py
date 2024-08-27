import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Start timing the program execution
start_time = time.time()

# Load the data
data = pd.read_excel(r'D:\srinidhi\amrita\MFC\SMOTE_analysis.xlsx')

# Convert DataFrame to NumPy arrays
X = data.iloc[:, :-1].values  # Features (embeddings)
y = data['Classification'].values  # Target

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Define the parameters for grid search
param_grid = {'weights': ['uniform', 'distance']}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X, y)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)

# Now, you can use the best parameters found by grid search for KNN
best_knn = KNeighborsClassifier(n_neighbors=5, **best_params)

# Perform cross-validation with the best parameters
cv_accuracy = cross_val_score(best_knn, X, y, cv=10, scoring='accuracy')
cv_precision = cross_val_score(best_knn, X, y, cv=10, scoring='precision_weighted')
cv_recall = cross_val_score(best_knn, X, y, cv=10, scoring='recall_weighted')
cv_f1 = cross_val_score(best_knn, X, y, cv=10, scoring='f1_weighted')

# Calculate mean and standard deviation for each metric
mean_accuracy = np.mean(cv_accuracy)
std_accuracy = np.std(cv_accuracy)

mean_precision = np.mean(cv_precision)
std_precision = np.std(cv_precision)

mean_recall = np.mean(cv_recall)
std_recall = np.std(cv_recall)

mean_f1 = np.mean(cv_f1)
std_f1 = np.std(cv_f1)

# Print the results
print("Precision mean: {:.2f}".format(mean_precision))
print("Precision std: {:.2f}".format(std_precision))

print("Recall mean: {:.2f}".format(mean_recall))
print("Recall std: {:.2f}".format(std_recall))

print("F1 Score mean: {:.2f}".format(mean_f1))
print("F1 Score std: {:.2f}".format(std_f1))

print("Accuracy mean: {:.2f}".format(mean_accuracy))
print("Accuracy std: {:.2f}".format(std_accuracy))

# Load testing data
testing_data = pd.read_excel(r'D:\srinidhi\amrita\MFC\sample_test.xlsx')

# Assuming you already have 'X_test' containing the features (embeddings) for testing
X_test = testing_data.values

# Predict classifications for testing data
predictions = grid_search.predict(X_test)

# Add predictions to the testing data DataFrame
testing_data['Predicted_Classification'] = predictions

# Save the testing data with predicted classifications to a new Excel file
testing_output_file = "predicted_testing_t5_knn.xlsx"
testing_data.to_excel(testing_output_file, index=False)
print("Predictions saved to:", testing_output_file)

# End timing the program execution
end_time = time.time()
execution_time = end_time - start_time

print("Time taken for running the whole program: {:.2f} seconds".format(execution_time))
