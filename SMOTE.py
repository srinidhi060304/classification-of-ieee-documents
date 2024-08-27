import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = r'D:\srinidhi\amrita\MFC\reduced_embeddings.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X = df.iloc[:, :-1]  # Features (embed_0 to embed_383)
y = df['Classification']  # Target

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Perform SMOTE resampling
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert resampled data to DataFrame
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['Classification'] = y_resampled

# Save resampled data to Excel file
resampled_file_path = r'D:\srinidhi\amrita\MFC\SMOTE_analysis.xlsx'
resampled_df.to_excel(resampled_file_path, index=False)

print("Resampled data saved to:", resampled_file_path)
