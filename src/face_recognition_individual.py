import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from utils.utils import create_submission, generate_statistics, plot_class_distribution
from sklearn.manifold import TSNE
from deepface import DeepFace
import time


class FaceRecognitionIndividual:
    def __init__(self, config=None):
        print("Initializing FaceRecognitionIndividual...")
        start_time = time.time()
        self.config = {
            'model_name': 'Facenet512',
            'backend': 'retinaface',
            'use_alignment': True,
            'similarity_threshold': 0.65,
            'random_seed': 42,
            'embeddings_dir': 'embeddings',
            'metrics_dir': 'metrics',
            'outputs_dir': 'outputs',
            'top_k': 5  # Number of most similar images to consider for voting
        }

        if config:
            self.config.update(config)

        for directory in [self.config['embeddings_dir'], self.config['outputs_dir'], self.config['metrics_dir']]:
            os.makedirs(directory, exist_ok=True)

        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])

        # {image_path: {'embedding': embedding, 'person_id': person_id}}
        self.train_embeddings_individual = {}
        self.test_embeddings = {}
        self.predictions = []
        self.class_counts = defaultdict(int)
        self.class_confidences = defaultdict(list)
        print(
            f"Initialization completed in {time.time() - start_time:.2f} seconds\n")

    def get_embedding(self, image_path):
        try:
            reps = DeepFace.represent(
                img_path=image_path,
                model_name=self.config['model_name'],
                detector_backend=self.config['backend'],
                align=self.config['use_alignment'],
                enforce_detection=False
            )
            if reps and 'embedding' in reps[0]:
                embedding = np.array(reps[0]['embedding'])
                return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        return None

    def load_embeddings(self, name="train_embeddings_individual"):
        model_name = self.config['model_name']
        embeddings_path = os.path.join(
            self.config['embeddings_dir'], model_name, f"{name}.npz")

        if not os.path.exists(embeddings_path):
            print(
                f"No saved embeddings found for {model_name} at {embeddings_path}")
            return {}

        try:
            data = np.load(embeddings_path, allow_pickle=True)
            embeddings = data['embeddings'].item()
            print(
                f"Loaded {len(embeddings)} embeddings for {model_name} from {name}")
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings for {model_name}: {e}")
            return {}

    def save_embeddings(self, embeddings, name="train_embeddings_individual"):
        model_name = self.config['model_name']
        model_dir = os.path.join(self.config['embeddings_dir'], model_name)
        os.makedirs(model_dir, exist_ok=True)

        save_path = os.path.join(model_dir, f"{name}.npz")
        np.savez_compressed(save_path, embeddings=embeddings)
        print(f"Saved {len(embeddings)} embeddings to {save_path}")
        return save_path

    def load_test_embeddings(self):
        self.test_embeddings = self.load_embeddings(name="test_embeddings")
        return self.test_embeddings

    def process_train_directory(self, train_dir):
        print(f"Processing training directory: {train_dir}")
        individual_embeddings = {}

        for person_dir in tqdm(os.listdir(train_dir), desc="Processing persons"):
            person_path = os.path.join(train_dir, person_dir)
            if not os.path.isdir(person_path):
                continue

            person_images = [os.path.join(person_path, f) for f in os.listdir(person_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not person_images:
                continue

            for img_path in person_images:
                embedding = self.get_embedding(img_path)
                if embedding is not None:
                    individual_embeddings[img_path] = {
                        'embedding': embedding,
                        'person_id': person_dir
                    }

        print(
            f"Generated individual embeddings for {len(individual_embeddings)} images")
        self.train_embeddings_individual = individual_embeddings
        self.save_embeddings(individual_embeddings,
                             name="train_embeddings_individual")
        return individual_embeddings

    def predict_person_by_voting(self, test_embedding, threshold=None):
        threshold = threshold if threshold is not None else self.config['similarity_threshold']
        top_k = self.config['top_k']

        if not self.train_embeddings_individual:
            return "doesn't_exist", 0.0

        # Extract all embeddings and their corresponding person IDs
        train_paths = list(self.train_embeddings_individual.keys())
        train_embeddings = np.vstack([self.train_embeddings_individual[path]['embedding'].reshape(1, -1)
                                     for path in train_paths])

        # Calculate similarity scores
        similarities = cosine_similarity(
            test_embedding.reshape(1, -1), train_embeddings)[0]

        # Find top K most similar images
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]

        # If the highest similarity is below threshold, return "doesn't_exist"
        if top_similarities[0] < threshold:
            return "doesn't_exist", top_similarities[0]

        # Get person IDs for top K similar images
        top_person_ids = [
            self.train_embeddings_individual[train_paths[idx]]['person_id'] for idx in top_indices]

        # Perform majority voting
        if not top_person_ids:
            return "doesn't_exist", 0.0

        person_votes = Counter(top_person_ids)
        most_common_person = person_votes.most_common(1)[0][0]

        # Calculate confidence as average similarity for the most common person
        person_indices = [i for i, idx in enumerate(top_indices)
                          if self.train_embeddings_individual[train_paths[idx]]['person_id'] == most_common_person]

        if not person_indices:
            return "doesn't_exist", 0.0

        person_similarities = [top_similarities[i] for i in person_indices]
        avg_confidence = sum(person_similarities) / len(person_similarities)

        return most_common_person, avg_confidence

    def process_images(self, images_dir, desc="Processing images"):
        result_embeddings = {}

        image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Found {len(image_files)} image files in {images_dir}")

        for img_path in tqdm(image_files, desc=desc):
            embedding = self.get_embedding(img_path)
            if embedding is not None:
                result_embeddings[img_path] = embedding

        print(
            f"Successfully processed {len(result_embeddings)}/{len(image_files)} images")
        return result_embeddings

    def process_test_images(self, test_dir):
        print(f"Processing test images from {test_dir}...")
        self.test_embeddings = self.process_images(
            test_dir, desc="Generating test embeddings")
        self.save_embeddings(self.test_embeddings, name="test_embeddings")
        return self.test_embeddings

    def generate_predictions(self, threshold=None):
        threshold = threshold if threshold is not None else self.config['similarity_threshold']

        if not self.train_embeddings_individual:
            print("No training embeddings loaded. Call load_embeddings() first.")
            return []

        if not self.test_embeddings:
            print("No test embeddings. Call process_test_images() first.")
            return []

        self.predictions = []
        self.class_counts = defaultdict(int)
        self.class_confidences = defaultdict(list)

        for img_path, emb in tqdm(self.test_embeddings.items(), desc="Generating predictions"):
            filename = os.path.basename(img_path)
            image_path = f"test/{filename}"

            prediction, confidence = self.predict_person_by_voting(
                emb, threshold)

            self.class_counts[prediction] += 1
            self.class_confidences[prediction].append(confidence)

            self.predictions.append({
                'gt': prediction,
                'image': image_path,
                'confidence': confidence
            })

        return self.predictions

    def test_on_images(self, test_dir="testing/"):
        print(f"Testing on images from {test_dir}...")

        if not os.path.exists(test_dir):
            print(f"Directory {test_dir} does not exist")
            return None

        if not self.train_embeddings_individual:
            self.load_embeddings()

        if not self.train_embeddings_individual:
            print("No training embeddings available. Cannot proceed with testing.")
            return None

        results = {}
        test_embeddings = self.process_images(test_dir, desc="Analyzing faces")

        for img_path, embedding in test_embeddings.items():
            prediction, confidence = self.predict_person_by_voting(embedding)

            results[img_path] = {
                'identity': prediction,
                'confidence': confidence
            }

            print(f"\nResults for {os.path.basename(img_path)}:")
            print(f"Predicted Identity: {prediction}")
            print(f"Confidence: {confidence:.4f}")

        return results

    def create_submission(self, threshold=None):
        return create_submission(self.predictions, self.config, threshold)

    def generate_statistics(self):
        return generate_statistics(self.predictions, self.class_counts, self.class_confidences, self.config)

    def plot_class_distribution(self):
        return plot_class_distribution(self.class_counts, self.class_confidences, self.config)

    def visualize_embeddings(self):
        if not self.train_embeddings_individual:
            print("No training embeddings available. Load embeddings first.")
            return

        # Extract embeddings and labels
        embeddings = []
        labels = []

        for img_path, data in self.train_embeddings_individual.items():
            embeddings.append(data['embedding'])
            labels.append(data['person_id'])

        if len(embeddings) == 0:
            print("No embeddings found to visualize.")
            return

        X = np.array(embeddings)
        y = np.array(labels)

        try:
            reducer = TSNE(
                n_components=2, random_state=self.config['random_seed'], perplexity=30)

            print("Fitting t-SNE...")
            X_reduced = reducer.fit_transform(X)

            plt.figure(figsize=(15, 10))
            scatter = plt.scatter(
                X_reduced[:, 0], X_reduced[:, 1],
                c=pd.factorize(y)[0],
                cmap='gist_ncar',
                alpha=0.6,
                s=10
            )
            plt.title(
                f't-SNE Visualization of Individual Face Embeddings\n({len(np.unique(y))} Classes)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')

            output_path = os.path.join(
                self.config['metrics_dir'], 'individual_embedding_tsne.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved visualization to {output_path}")

        except Exception as e:
            print(f"Error during visualization: {e}")

    def run_pipeline(self, train_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/train',
                     test_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/test'):
        total_start_time = time.time()
        print("\n" + "="*50)
        print("Starting Face Recognition Pipeline")
        print("="*50)
        print(f"Model: {self.config['model_name']}")
        print(f"Similarity threshold: {self.config['similarity_threshold']}")
        print(f"Top-{self.config['top_k']} voting system")
        print("-"*50 + "\n")

        # Loading/Processing training embeddings
        train_start = time.time()
        print("Step 1: Loading training embeddings...")
        self.train_embeddings_individual = self.load_embeddings()
        if not self.train_embeddings_individual:
            print("No cached embeddings found, processing training directory...")
            self.process_train_directory(train_dir)
        print(
            f"Training embeddings ready. Time taken: {time.time() - train_start:.2f} seconds\n")

        # Loading/Processing test embeddings
        test_start = time.time()
        print("Step 2: Loading test embeddings...")
        self.test_embeddings = self.load_test_embeddings()
        if not self.test_embeddings:
            print("No cached test embeddings found, processing test directory...")
            self.process_test_images(test_dir)
        print(
            f"Test embeddings ready. Time taken: {time.time() - test_start:.2f} seconds\n")

        # Generating predictions
        pred_start = time.time()
        print("Step 3: Generating predictions...")
        self.generate_predictions()
        print(
            f"Predictions generated. Time taken: {time.time() - pred_start:.2f} seconds\n")

        # Creating submission
        sub_start = time.time()
        print("Step 4: Creating submission file...")
        submission_path = self.create_submission()
        print(
            f"Submission created. Time taken: {time.time() - sub_start:.2f} seconds\n")

        # Generating statistics and visualizations
        viz_start = time.time()
        print("Step 5: Generating statistics and visualizations...")
        self.generate_statistics()
        self.plot_class_distribution()
        self.visualize_embeddings()
        print(
            f"Visualizations completed. Time taken: {time.time() - viz_start:.2f} seconds\n")

        print("="*50)
        print(
            f"Pipeline completed! Total time: {time.time() - total_start_time:.2f} seconds")
        print("="*50)

        return submission_path






if __name__ == "__main__":
    face_recognition = FaceRecognitionIndividual()

    submission_path = face_recognition.run_pipeline(
        train_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/train',
        test_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/test'
    )
    print(f"Submission file created at: {submission_path}")
