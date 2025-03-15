        
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import faiss
from tqdm import tqdm
from deepface import DeepFace
from collections import defaultdict
from utils import create_submission, generate_statistics, plot_class_distribution, plot_confusion_matrix, organize_by_predictions
from sklearn.manifold import TSNE


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
        self.faiss_index = None
        self.person_ids = []
        self.test_embeddings = {}
        self.predictions = []
        self.class_counts = defaultdict(int)
        self.class_confidences = defaultdict(list)
        self.embedding_dim = 512  # Default for Facenet512

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
        index_path = os.path.join(
            self.config['embeddings_dir'], model_name, f"{name}_faiss.index")
        person_ids_path = os.path.join(
            self.config['embeddings_dir'], model_name, f"{name}_person_ids.npy")

        if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
            print(
                f"No saved embeddings found for {model_name} at {embeddings_path}")
            return {}

        try:
            # Load the original embeddings dictionary
            data = np.load(embeddings_path, allow_pickle=True)
            embeddings = data['embeddings'].item()

            # Load the FAISS index
            self.faiss_index = faiss.read_index(index_path)

            # Load the person_ids list
            self.person_ids = np.load(
                person_ids_path, allow_pickle=True).tolist()

            print(
                f"Loaded {len(embeddings)} embeddings for {model_name} from {name}")
            print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

            if name == "train_embeddings":
                self.train_embeddings = embeddings

            return embeddings
        except Exception as e:
            print(f"Error loading embeddings for {model_name}: {e}")
            return {}

    def save_embeddings(self, embeddings, name="train_embeddings"):
        model_name = self.config['model_name']
        model_dir = os.path.join(self.config['embeddings_dir'], model_name)
        os.makedirs(model_dir, exist_ok=True)

        save_path = os.path.join(model_dir, f"{name}.npz")
        index_path = os.path.join(model_dir, f"{name}_faiss.index")
        person_ids_path = os.path.join(model_dir, f"{name}_person_ids.npy")

        # Save the original embeddings dictionary
        np.savez_compressed(save_path, embeddings=embeddings)

        # If this is for training data, create and save the FAISS index
        if name == "train_embeddings":
            persons = list(embeddings.keys())
            embeddings_matrix = np.vstack([embeddings[p] for p in persons])

            # Get dimension from the first embedding
            self.embedding_dim = embeddings_matrix.shape[1]

            # Create and train a FAISS index for these embeddings
            # Using InnerProduct since we have normalized vectors
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(embeddings_matrix.astype('float32'))

            # Save the FAISS index
            faiss.write_index(self.faiss_index, index_path)

            # Save the person IDs in the same order as they were added to the index
            self.person_ids = persons
            np.save(person_ids_path, persons)

        print(f"Saved {len(embeddings)} embeddings to {save_path}")
        if name == "train_embeddings":
            print(
                f"Saved FAISS index with {self.faiss_index.ntotal} vectors to {index_path}")

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

        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return [("doesn't_exist", 0.0)] if top_n > 1 else ("doesn't_exist", 0.0)

        # Make sure the test embedding is a properly formatted numpy array
        query_embedding = test_emb.reshape(1, -1).astype('float32')

        # Search the FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_n)

        # FAISS with IndexFlatIP returns similarities (higher is better since we use inner product)
        # but to maintain compatibility with the original code, we ensure these are treated as similarities
        similarities = distances[0]  # Get the first query's results
        indices = indices[0]  # Get the first query's results

        # Convert the index results back to person IDs
        top_matches = []
        for i, sim in zip(indices, similarities):
            if i < len(self.person_ids):
                person_id = self.person_ids[i]
                top_matches.append((person_id, float(sim)))
            else:
                top_matches.append(("doesn't_exist", 0.0))

        if top_n == 1:
            if not top_matches:
                return ("doesn't_exist", 0.0)
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

        if not self.faiss_index or self.faiss_index.ntotal == 0:
            print("No FAISS index available. Call load_embeddings() first.")
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

        if not self.faiss_index or self.faiss_index.ntotal == 0:
            self.load_embeddings()

        if not self.faiss_index or self.faiss_index.ntotal == 0:
            print("No FAISS index available. Cannot proceed with testing.")
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

    def plot_confusion_matrix(self, results_df, output_dir, top_n=20):
        return plot_confusion_matrix(results_df, output_dir, top_n)

    def organize_by_predictions(self, test_dir, output_dir=None):
        return organize_by_predictions(test_dir, self.train_embeddings, self.get_embedding,
                                       self.predict_person, self.config, output_dir)

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
        if not self.train_embeddings or not self.faiss_index or self.faiss_index.ntotal == 0:
            self.process_train_directory(train_dir)

        self.test_embeddings = self.load_test_embeddings()
        if not self.test_embeddings:
            self.process_test_images(test_dir)

        self.generate_predictions()
        submission_path = self.create_submission()
        self.generate_statistics()
        self.plot_class_distribution()
        self.visualize_embeddings()

        return submission_path