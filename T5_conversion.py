import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the dataset
file_path = r'D:\srinidhi\amrita\MFC\ML_Methodology.xlsx'
df = pd.read_excel(file_path)

# Define the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/sentence-t5-base')

# Function to encode equations into embeddings
def encode_equation(Methodology):
    return model.encode(str(Methodology))

# Apply encoding to the "Equation" column
df['EmbeddingsLM'] = df['Methodology'].apply(encode_equation)

# Create DataFrame with embeddings
t5_embeddings = pd.DataFrame(df['EmbeddingsLM'].tolist(), index=df.index).add_prefix('embed_')

# Optionally, you can save the embeddings DataFrame to a new Excel file
output_file_path = r'D:\srinidhi\amrita\MFC\ML_t5_Methodology.xlsx'
t5_embeddings.to_excel(output_file_path, index=False)

# Print a message indicating completion
print("T5 embeddings have been generated and saved to:", output_file_path)
