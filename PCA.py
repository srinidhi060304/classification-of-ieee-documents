import pandas as pd
from sklearn.decomposition import PCA

# Load the embeddings DataFrame with 768 dimensions
embeddings_df = pd.read_excel(r"D:\srinidhi\amrita\MFC\ML_t5_Methodology.xlsx")

# Get the minimum between the number of samples and features
min_components = min(embeddings_df.shape)

# Initialize PCA with the maximum feasible number of components
pca = PCA(n_components=min_components)

# Fit PCA to the embeddings data and transform it to the reduced dimensionality
embeddings_reduced = pca.fit_transform(embeddings_df)

# Create DataFrame with reduced dimension embeddings
embeddings_reduced_df = pd.DataFrame(embeddings_reduced, index=embeddings_df.index)

# Optionally, you can save the reduced dimension embeddings DataFrame to a new Excel file
output_file_path = "reduced_embeddings.xlsx"
embeddings_reduced_df.to_excel(output_file_path, index=False)

# Print a message indicating completion
print("Embeddings with reduced dimensionality have been generated and saved to:", output_file_path)
