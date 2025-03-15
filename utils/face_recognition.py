import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
import seaborn as sns

"""
FaceRecognition: A comprehensive face recognition system using DeepFace embeddings.
- Computes and manages facial embeddings for known identities
- Processes test images and predicts identities based on facial similarity
- Provides visualization, metrics calculation, and organization of results
- Default model: Facenet512 with RetinaFace backend
- Customizable similarity threshold (default: 0.65)
- Supports evaluation on labeled datasets and organizing predictions
- Modified testing function now processes images from a 'testing/' directory
  and displays top 3 most similar identities with confidence scores
"""
"""
Future Improvements:
- enhance the performance of the face recognition system using batching and multi-threading.
- add ensemble methods to combine predictions from multiple models.
- add support for multiple distance metrics and similarity thresholds.
- visualize the embeddings using dimensionality reduction techniques.
- data augmentation and fine-tuning for improved embeddings.

- add feature for single embedding ( using Vector DB) and classify using most 3 similar embeddings.
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

        os.makedirs(self.config['embeddings_dir'], exist_ok=True)
        os.makedirs(self.config['outputs_dir'], exist_ok=True)
        os.makedirs(self.config['metrics_dir'], exist_ok=True)

        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['random_seed'])
            print("GPU is available and will be used")
        else:
            print("GPU not available, using CPU")

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
                normalized_embedding = embedding / np.linalg.norm(embedding)
                return normalized_embedding
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        return None

    def load_embeddings(self):
        model_name = self.config['model_name']
        embeddings_path = os.path.join(
            self.config['embeddings_dir'], model_name, "train_embeddings.npz")

        if not os.path.exists(embeddings_path):
            print(
                f"No saved embeddings found for {model_name} at {embeddings_path}")
            return {}

        try:
            data = np.load(embeddings_path, allow_pickle=True)
            self.train_embeddings = data['embeddings'].item()

            normalized_embeddings = {}
            for person, emb in self.train_embeddings.items():
                norm = np.linalg.norm(emb)
                if norm > 0:
                    normalized_embeddings[person] = emb / norm
                else:
                    normalized_embeddings[person] = emb

            print(
                f"Loaded and normalized {len(normalized_embeddings)} embeddings for {model_name}")
            self.train_embeddings = normalized_embeddings
            return normalized_embeddings
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

    def predict_person(self, test_emb, threshold=None):
        if threshold is None:
            threshold = self.config['similarity_threshold']

        if not self.train_embeddings:
            return "doesn't_exist", 0.0

        persons = list(self.train_embeddings.keys())
        embeddings = np.vstack(
            [self.train_embeddings[p].reshape(1, -1) for p in persons])

        sims = cosine_similarity(test_emb.reshape(1, -1), embeddings)[0]

        if len(sims) == 0:
            return "doesn't_exist", 0.0

        max_idx = np.argmax(sims)
        max_sim = sims[max_idx]
        pred_person = persons[max_idx]

        return (pred_person if max_sim >= threshold else "doesn't_exist"), max_sim

    def predict_top_persons(self, test_emb, top_n=3):
        if not self.train_embeddings:
            return [("doesn't_exist", 0.0)]

        persons = list(self.train_embeddings.keys())
        embeddings = np.vstack(
            [self.train_embeddings[p].reshape(1, -1) for p in persons])

        sims = cosine_similarity(test_emb.reshape(1, -1), embeddings)[0]

        if len(sims) == 0:
            return [("doesn't_exist", 0.0)]

        top_indices = np.argsort(sims)[::-1][:top_n]
        return [(persons[idx], sims[idx]) for idx in top_indices]

    def process_test_images(self, test_dir):
        print(f"Processing test images from {test_dir}...")

        test_files = []
        for filename in os.listdir(test_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                test_files.append(os.path.join(test_dir, filename))

        print(f"Found {len(test_files)} test files.")

        for img_path in tqdm(test_files, desc="Generating embeddings"):
            embedding = self.get_embedding(img_path)
            if embedding is not None:
                self.test_embeddings[img_path] = embedding

        print(
            f"Successfully processed {len(self.test_embeddings)}/{len(test_files)} test images")
        return self.test_embeddings

    def generate_predictions(self, threshold=None):
        if threshold is None:
            threshold = self.config['similarity_threshold']

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

    def create_submission(self, threshold=None):
        if threshold is None:
            threshold = self.config['similarity_threshold']

        if not self.predictions:
            print("No predictions available. Call generate_predictions() first.")
            return None

        submission_df = pd.DataFrame(self.predictions)
        submission_df['frame'] = -1
        submission_df['objects'] = submission_df.apply(
            lambda row: {
                'gt': row['gt'],
                'image': f"test_set/{os.path.basename(row['image'])}",
                'confidence': row['confidence']
            },
            axis=1
        )
        submission_df['objective'] = 'face_reid'
        submission_df.drop(columns=['gt', 'image', 'confidence'], inplace=True)

        model_name = self.config['model_name']
        index = 0
        while os.path.exists(f'{self.config["outputs_dir"]}/submission_{model_name}_{threshold:.2f}_{index}.csv'):
            index += 1

        submission_path = f'{self.config["outputs_dir"]}/submission_{model_name}_{threshold:.2f}_{index}.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        return submission_path

    def generate_statistics(self):
        if not self.predictions:
            print("No predictions available. Call generate_predictions() first.")
            return None, None

        confidence_means = {}
        for cls, confidences in self.class_confidences.items():
            confidence_means[cls] = np.mean(confidences)

        print("\nClass Distribution Summary:")
        print(f"Total unique classes predicted: {len(self.class_counts)}")
        print(
            f"Most common class: {max(self.class_counts.items(), key=lambda x: x[1])[0]}")
        print("Number of 'doesn't_exist' predictions: {}".format(
            self.class_counts.get("doesn't_exist", 0)))

        class_stats = []
        for cls, count in self.class_counts.items():
            class_stats.append({
                'class': cls,
                'count': count,
                'mean_confidence': confidence_means[cls]
            })

        model_name = self.config['model_name']
        threshold = self.config['similarity_threshold']

        class_stats_df = pd.DataFrame(class_stats)
        class_stats_df = class_stats_df.sort_values('count', ascending=False)
        stats_path = f'{self.config["outputs_dir"]}/class_stats_{model_name}_{threshold:.2f}.csv'
        class_stats_df.to_csv(stats_path, index=False)
        print(f"Detailed class statistics saved to {stats_path}")

        return class_stats_df, confidence_means

    def plot_class_distribution(self):
        if not self.class_counts:
            print("No class predictions available. Generate predictions first.")
            return

        confidence_means = {}
        for cls, confidences in self.class_confidences.items():
            confidence_means[cls] = np.mean(confidences)

        plt.figure(figsize=(14, 8))

        sorted_items = sorted(self.class_counts.items(),
                              key=lambda x: x[1], reverse=True)
        classes = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        if len(classes) > 30:
            classes = classes[:30]
            counts = counts[:30]
            plt.title('Top 30 Predicted Classes Distribution')
        else:
            plt.title('Predicted Classes Distribution')

        bars = plt.bar(classes, counts, color='skyblue')

        for i, (cls, count) in enumerate(zip(classes, counts)):
            if cls in confidence_means:
                conf = confidence_means[cls]
                plt.text(i, count + 0.5, f"{conf:.2f}",
                         ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Predicted Class')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.figtext(0.91, 0.85, 'Mean Confidence', fontweight='bold')

        plt.tight_layout()

        plt.savefig(
            f'{self.config["outputs_dir"]}/class_distribution.png', dpi=300, bbox_inches='tight')
        print(
            f"Class distribution plot saved to {self.config['outputs_dir']}/class_distribution.png")

        plt.show()

    def run_pipeline(self, test_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/test'):
        print("Initializing face recognition pipeline...")
        print(f"Using model: {self.config['model_name']}")
        print(
            f"Using similarity threshold: {self.config['similarity_threshold']}")

        self.load_embeddings()
        self.process_test_images(test_dir)
        self.generate_predictions()
        submission_path = self.create_submission()
        self.generate_statistics()
        self.plot_class_distribution()

        return submission_path

    def test_on_images(self, test_dir="testing/"):
        """
        Process images in the testing directory and show top matching identities.

        Args:
            test_dir (str): Directory containing test images, default is 'testing/'

        Returns:
            dict: Dictionary with results for each image
        """
        print(f"Testing on images from {test_dir}...")

        # Ensure directory exists
        if not os.path.exists(test_dir):
            print(f"Directory {test_dir} does not exist")
            return None

        # Load embeddings if not already loaded
        if not self.train_embeddings:
            self.load_embeddings()

        if not self.train_embeddings:
            print("No training embeddings available. Cannot proceed with testing.")
            return None

        # Find all image files
        test_files = []
        for filename in os.listdir(test_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                test_files.append(os.path.join(test_dir, filename))

        if not test_files:
            print(f"No image files found in {test_dir}")
            return None

        print(f"Found {len(test_files)} image files to process")

        # Process each image and collect results
        results = {}

        for img_path in tqdm(test_files, desc="Analyzing faces"):
            embedding = self.get_embedding(img_path)

            if embedding is not None:
                # Get top prediction
                top_pred, top_conf = self.predict_person(embedding)

                # Get top 3 most similar identities
                top_matches = self.predict_top_persons(embedding, top_n=3)

                # Store results
                results[img_path] = {
                    'identity': top_pred,
                    'confidence': top_conf,
                    'top_matches': top_matches
                }

                # Print results for this image
                print(f"\nResults for {os.path.basename(img_path)}:")
                print(f"Predicted Identity: {top_pred}")
                print(f"Confidence: {top_conf:.4f}")
                print("Top 3 Matches:")
                for i, (person, conf) in enumerate(top_matches, 1):
                    print(f"  {i}. {person} ({conf:.4f})")
            else:
                print(f"Could not extract embedding from {img_path}")

        return results

    def _plot_confusion_matrix(self, results_df, output_dir, top_n=20):
        top_classes = results_df['true_label'].value_counts().head(
            top_n).index.tolist()

        filtered_df = results_df[results_df['true_label'].isin(top_classes)]

        conf_matrix = pd.crosstab(
            filtered_df['true_label'],
            filtered_df['pred_label'],
            normalize='index'
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=True, yticklabels=True)
        plt.title('Normalized Confusion Matrix (Top Classes)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {matrix_path}")
        plt.close()

    def organize_by_predictions(self, test_dir, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(
                self.config['outputs_dir'], 'organized_predictions')

        print(f"Organizing test images by predictions...")

        os.makedirs(output_dir, exist_ok=True)

        if not self.train_embeddings:
            self.load_embeddings()

        for filename in tqdm(os.listdir(test_dir), desc="Organizing images"):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(test_dir, filename)

                embedding = self.get_embedding(img_path)
                if embedding is not None:
                    pred_label, confidence = self.predict_person(embedding)

                    pred_dir = os.path.join(output_dir, pred_label)
                    os.makedirs(pred_dir, exist_ok=True)

                    dest_filename = f"{confidence:.4f}_{filename}"
                    shutil.copy2(img_path, os.path.join(
                        pred_dir, dest_filename))

        print(f"Images organized by predictions in {output_dir}")
        return output_dir
