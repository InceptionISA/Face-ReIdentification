import numpy as np
import pandas as pd
import os
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load the saved embeddings
print("Loading saved embeddings...")
embeddings_df = pd.read_csv('embeddings/facenet.csv')

# Convert the embedding dataframe format to a dictionary of person_id -> embedding
train_embeddings = {}
for column in embeddings_df.columns:
    if column.startswith('person_'):
        # Get the embedding for this person (all rows in this column)
        embedding = embeddings_df[column].values
        # Only use the embedding if it's not NaN or empty
        if not np.isnan(embedding).any():
            train_embeddings[column] = embedding

print(f"Loaded embeddings for {len(train_embeddings)} people")

# Function to compute embedding using DeepFace


def get_embedding_deepface(image_path, model_name='Facenet'):
    try:
        reps = DeepFace.represent(
            img_path=image_path, model_name=model_name, enforce_detection=False)
        if reps and 'embedding' in reps[0]:
            return np.array(reps[0]['embedding'])
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    return None

# Predict identity using cosine similarity


def predict_person(test_emb, threshold=0.6):
    similarities = {person: cosine_similarity(test_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
                    for person, emb in train_embeddings.items()}
    pred_person, max_sim = max(
        similarities.items(), key=lambda x: x[1], default=(None, -1))
    return pred_person if max_sim >= threshold else "doesn't_exist"


# Get all test files
print("Getting test files...")
test_dir = '/kaggle/input/surveillance-for-retail-stores/face_identification/face_identification/test/'
test_files = []
for filename in os.listdir(test_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        test_files.append(os.path.join(test_dir, filename))

print(f"Found {len(test_files)} test files.")

# Make predictions for test files
print("Making predictions for test files...")
results = []
for img_path in tqdm(test_files, desc="Processing test files"):
    filename = os.path.basename(img_path)
    # Ensure the image path is in the format 'test/x.jpg'
    image_path = f"test/{filename}"

    emb = get_embedding_deepface(img_path)
    if emb is not None:
        prediction = predict_person(emb)
    else:
        prediction = "doesn't_exist"  # Default if embedding fails

    # Match the required output format
    results.append({'gt': prediction, 'image': image_path})

# Create submission DataFrame
submission_df = pd.DataFrame(results)
submission_df['frame'] = -1

submission_df['objects'] = submission_df.apply(lambda row: {
                             'gt': row['gt'], 'image': 'test_set/{}'.format(row['image'].split('/')[-1])}, axis=1)
submission_df['objective'] = 'face_reid'
submission_df.drop(columns=['gt', 'image'], inplace=True)

print(submission_df.head())

os.makedirs('outputs', exist_ok=True)

# Determine the next available index for the submission file
index = 0
while os.path.exists(f'outputs/submission{index}.csv'):
    index += 1

# Save submission to CSV with incremented index
submission_path = f'outputs/submission{index}.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
