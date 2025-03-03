import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
MODELS = [
        # 'Facenet',
        'Facenet512',
        #    'VGG-Face', 'Dlib'
        ]

BACKEND = 'retinaface'
USE_ALIGNMENT = True
SIMILARITY_THRESHOLD = 0.3
RANDOM_SEED = 42
EMBEDDINGS_DIR = 'embeddings'

# Ensure embeddings directory exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)


def get_embedding_deepface(image_path, model_name='Facenet512'):
    """Compute embedding using DeepFace with specified model and RetinaFace backend"""
    try:
        reps = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=BACKEND,
            align=USE_ALIGNMENT,
            enforce_detection=False
        )
        if reps and 'embedding' in reps[0]:
            return np.array(reps[0]['embedding'])
    except Exception as e:
        print(f"Error processing {image_path} with {model_name}: {e}")
    return None


def load_embeddings(model_name='Facenet512'):
    """Load saved embeddings for the specified model"""
    embeddings_path = os.path.join(
        EMBEDDINGS_DIR, model_name, "train_embeddings.npz")

    if not os.path.exists(embeddings_path):
        print(
            f"No saved embeddings found for {model_name} at {embeddings_path}")
        return {}

    try:
        data = np.load(embeddings_path, allow_pickle=True)
        train_embeddings = data['embeddings'].item()
        print(f"Loaded {len(train_embeddings)} embeddings for {model_name}")
        return train_embeddings
    except Exception as e:
        print(f"Error loading embeddings for {model_name}: {e}")
        return {}


def predict_person(test_emb, train_embeddings, threshold=SIMILARITY_THRESHOLD):
    """Predict identity using cosine similarity"""
    if not train_embeddings:
        return "doesn't_exist"

    similarities = {
        person: cosine_similarity(test_emb.reshape(
            1, -1), emb.reshape(1, -1))[0][0]
        for person, emb in train_embeddings.items()
    }

    if not similarities:
        return "doesn't_exist"

    pred_person, max_sim = max(similarities.items(), key=lambda x: x[1])
    # Strip 'person_' prefix if needed
    if pred_person.startswith('person_'):
        pred_person = pred_person
    return pred_person if max_sim >= threshold else "doesn't_exist"


def generate_submission(test_dir, model_name='Facenet512'):
    """Generate submission file for test images using the specified model"""
    print(f"Generating submission using {model_name} model...")

    # Load embeddings for the selected model
    train_embeddings = load_embeddings(model_name)

    if not train_embeddings:
        print(
            f"No embeddings available for {model_name}. Cannot generate predictions.")
        return

    # Get all test files
    test_files = []
    for filename in os.listdir(test_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            test_files.append(os.path.join(test_dir, filename))

    print(f"Found {len(test_files)} test files.")

    # Make predictions for test files
    results = []
    for img_path in tqdm(test_files, desc=f"Processing test files with {model_name}"):
        filename = os.path.basename(img_path)
        # Format image path as required by the submission format
        image_path = f"test/{filename}"

        # Get embedding for test image
        emb = get_embedding_deepface(img_path, model_name=model_name)

        if emb is not None:
            prediction = predict_person(emb, train_embeddings)
        else:
            prediction = "doesn't_exist"  # Default if embedding fails

        # Match the required output format
        results.append({'gt': prediction, 'image': image_path})

    # Create submission DataFrame with the right format
    submission_df = pd.DataFrame(results)
    submission_df['frame'] = -1
    submission_df['objects'] = submission_df.apply(
        lambda row: {
            'gt': row['gt'], 'image': f"test_set/{os.path.basename(row['image'])}"},
        axis=1
    )
    submission_df['objective'] = 'face_reid'
    submission_df.drop(columns=['gt', 'image'], inplace=True)

    # Determine the next available index for the submission file
    index = 0
    while os.path.exists(f'outputs/submission_{model_name}_{index}.csv'):
        index += 1

    # Save submission to CSV
    submission_path = f'outputs/submission_{model_name}_{index}.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return submission_path


def run_all_models(test_dir):
    """Generate submissions using all models"""
    all_submissions = {}

    for model_name in MODELS:
        print(f"\n{'='*50}")
        print(f"Processing model: {model_name}")
        print(f"{'='*50}")

        submission_path = generate_submission(test_dir, model_name)
        all_submissions[model_name] = submission_path

    print("\nAll submissions generated:")
    for model_name, path in all_submissions.items():
        print(f"{model_name}: {path}")

    return all_submissions


# Main execution
if __name__ == "__main__":
    # Path to test directory
    test_dir = '/kaggle/input/surveillance-for-retail-stores/face_identification/face_identification/test/'

    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        print("Please provide the correct path to the test directory.")
    else:
        print("Initializing face recognition submission pipeline...")
        all_submissions = run_all_models(test_dir)
        print("Process completed successfully!")