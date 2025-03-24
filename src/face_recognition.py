import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from utils.utils import create_submission, generate_statistics, plot_class_distribution
from sklearn.manifold import TSNE

"""
workspace  structure: 
tree
(base) PS E:\99\Competitions\Fawry\Face-ReIdentification> tree
E:.
├───dataset
│   └───surveillance-for-retail-stores
│       └───face_identification
│           └───face_identification
│               ├───test
│               │   ├───image1.jpg
│               │   ├───image2.jpg
│               │   ├───...
│               └───train
│                   ├───person_0
                        ├───image1.jpg
                        ├───image2.jpg
                        ├───...
│                   ├───person_1
│                   ├───...
│                   └───person_125
├───embeddings
│   └───Facenet512
│       ├───test_embeddings.npz
│       └───train_embeddings.npz
├───testing
│   ├───image1.jpg
│   ├───image2.jpg
│   ├───...
└───src
    ├───face_recognition.py
    └───face_recognition_individual.py
└───utils
    ├───utils.py
"""


class FaceRecognition:
    def __init__(self, config=None):
        self.config = {
            'model_name': 'Facenet512',
            'backend': 'retinaface',
            'use_alignment': True,
            'similarity_threshold': 0.65,
            'random_seed': 42,
            'embeddings_dir': 'embeddings',
            'metrics_dir': 'metrics',
            'outputs_dir': 'outputs'
        }

        if config:
            self.config.update(config)

        for directory in [self.config['embeddings_dir'], self.config['outputs_dir'], self.config['metrics_dir']]:
            os.makedirs(directory, exist_ok=True)

        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])

        self.train_embeddings = {}
        self.test_embeddings = {}
        self.predictions = []
        self.class_counts = defaultdict(int)
        self.class_confidences = defaultdict(list)

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

    def load_embeddings(self, name="train_embeddings"):
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

    def save_embeddings(self, embeddings, name="train_embeddings"):
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
        person_embeddings = {}

        for person_dir in tqdm(os.listdir(train_dir), desc="Processing persons"):
            person_path = os.path.join(train_dir, person_dir)
            if not os.path.isdir(person_path):
                continue

            person_images = [os.path.join(person_path, f) for f in os.listdir(person_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not person_images:
                continue

            all_embeddings = []

            for img_path in person_images:
                embedding = self.get_embedding(img_path)
                if embedding is not None:
                    all_embeddings.append(embedding)

            if all_embeddings:
                mean_embedding = np.mean(all_embeddings, axis=0)
                mean_embedding = mean_embedding / \
                    np.linalg.norm(mean_embedding)
                person_embeddings[person_dir] = mean_embedding

        print(f"Generated embeddings for {len(person_embeddings)} persons")
        self.train_embeddings = person_embeddings
        self.save_embeddings(person_embeddings, name="train_embeddings")
        return person_embeddings

    def predict(self, test_emb, top_n=1, threshold=None):
        threshold = threshold if threshold is not None else self.config['similarity_threshold']

        if not self.train_embeddings:
            return [("doesn't_exist", 0.0)] if top_n > 1 else ("doesn't_exist", 0.0)

        persons = list(self.train_embeddings.keys())
        embeddings = np.vstack(
            [self.train_embeddings[p].reshape(1, -1) for p in persons])
        sims = cosine_similarity(test_emb.reshape(1, -1), embeddings)[0]

        if len(sims) == 0:
            return [("doesn't_exist", 0.0)] if top_n > 1 else ("doesn't_exist", 0.0)

        top_indices = np.argsort(sims)[::-1][:top_n]
        top_matches = [(persons[idx], sims[idx]) for idx in top_indices]

        if top_n == 1:
            pred_person, max_sim = top_matches[0]
            return (pred_person if max_sim >= threshold else "doesn't_exist"), max_sim

        return top_matches

    def predict_person(self, embedding, threshold=None):
        return self.predict(embedding, top_n=1, threshold=threshold)

    def predict_top_persons(self, embedding, top_n=3):
        return self.predict(embedding, top_n=top_n)

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

        if not self.train_embeddings:
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

            prediction, confidence = self.predict_person(emb, threshold)

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

        if not self.train_embeddings:
            self.load_embeddings()

        if not self.train_embeddings:
            print("No training embeddings available. Cannot proceed with testing.")
            return None

        results = {}
        test_embeddings = self.process_images(test_dir, desc="Analyzing faces")

        for img_path, embedding in test_embeddings.items():
            top_pred, top_conf = self.predict_person(embedding)
            top_matches = self.predict_top_persons(embedding, top_n=3)

            results[img_path] = {
                'identity': top_pred,
                'confidence': top_conf,
                'top_matches': top_matches
            }

            print(f"\nResults for {os.path.basename(img_path)}:")
            print(f"Predicted Identity: {top_pred}")
            print(f"Confidence: {top_conf:.4f}")
            print("Top 3 Matches:")
            for i, (person, conf) in enumerate(top_matches, 1):
                print(f"  {i}. {person} ({conf:.4f})")

        return results

    def create_submission(self, threshold=None):
        return create_submission(self.predictions, self.config, threshold)

    def generate_statistics(self):
        return generate_statistics(self.predictions, self.class_counts, self.class_confidences, self.config)

    def plot_class_distribution(self):
        return plot_class_distribution(self.class_counts, self.class_confidences, self.config)

    def visualize_embeddings(self):
        if not self.train_embeddings:
            print("No training embeddings available. Load embeddings first.")
            return

        embeddings = []
        labels = []
        for person_name, emb in self.train_embeddings.items():
            embeddings.append(emb)
            labels.append(person_name)

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
                f't-SNE Visualization of Face Embeddings\n({len(np.unique(y))} Classes)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')

            output_path = os.path.join(
                self.config['metrics_dir'], 'embedding_tsne.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved visualization to {output_path}")

        except Exception as e:
            print(f"Error during visualization: {e}")

    def run_pipeline(self, train_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/train',
                     test_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/test'):
        print(
            f"Initializing face recognition pipeline with model: {self.config['model_name']}")
        print(
            f"Using similarity threshold: {self.config['similarity_threshold']}")

        self.train_embeddings = self.load_embeddings()
        if not self.train_embeddings:
            self.process_train_directory(train_dir)

        self.test_embeddings = self.load_test_embeddings()
        if not self.test_embeddings:
            self.process_test_images(test_dir)

        self.generate_predictions()
        submission_path = self.create_submission()
        self.generate_statistics()
        self.plot_class_distribution()
        # self.visualize_embeddings()

        return submission_path


if __name__ == "__main__":
    config = {
        'model_name': 'Facenet512',
        'backend': 'mtcnn',
        'similarity_threshold': 0.7677,
    }
    face_recognition = FaceRecognition(config)
    submission_path = face_recognition.run_pipeline()
    print(f"Pipeline completed. Submission saved to: {submission_path}")
