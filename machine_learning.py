import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Load preprocessed data
user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0)

# Perform matrix factorization
svd = TruncatedSVD(n_components=50)
latent_matrix = svd.fit_transform(user_item_matrix)

# Save the latent matrix for future use
pd.DataFrame(latent_matrix).to_csv('latent_matrix.csv', index=False)
