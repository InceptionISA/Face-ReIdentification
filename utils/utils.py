import shutil
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_submission(predictions, config, threshold=None):
    if threshold is None:
        threshold = config['similarity_threshold']

    if not predictions:
        print("No predictions available. Call generate_predictions() first.")
        return None

    submission_df = pd.DataFrame(predictions)
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

    model_name = config['model_name']
    index = 0
    submission_path = f'{config["outputs_dir"]}/submission_{model_name}_{threshold:.2f}_{index}.csv'

    while os.path.exists(submission_path):
        index += 1
        submission_path = f'{config["outputs_dir"]}/submission_{model_name}_{threshold:.2f}_{index}.csv'

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return submission_path


def generate_statistics(predictions, class_counts, class_confidences, config):
    if not predictions:
        print("No predictions available. Call generate_predictions() first.")
        return None, None

    # Calculate mean confidence for each class
    confidence_means = {cls: np.mean(confidences)
                                     for cls, confidences in class_confidences.items()}

    # Print summary statistics
    print("\nClass Distribution Summary:")
    print(f"Total unique classes predicted: {len(class_counts)}")

    if class_counts:
        most_common = max(class_counts.items(), key=lambda x: x[1])[0]
        print(f"Most common class: {most_common}")


    print(f"Number of doesnt_exist predictions:",{class_counts.get("doesn't_exist", 0)})

    # Create detailed class statistics dataframe
    class_stats = [
        {
            'class': cls,
            'count': count,
            'mean_confidence': confidence_means[cls]
        } 
        for cls, count in class_counts.items()
    ]

    class_stats_df = pd.DataFrame(class_stats).sort_values('count', ascending=False)
    
    # Save statistics to file
    model_name = config['model_name']
    threshold = config['similarity_threshold']
    stats_path = f'{config["outputs_dir"]}/class_stats_{model_name}_{threshold:.2f}.csv'
    class_stats_df.to_csv(stats_path, index=False)
    print(f"Detailed class statistics saved to {stats_path}")

    return class_stats_df, confidence_means


def plot_class_distribution(class_counts, class_confidences, config):
    if not class_counts:
        print("No class predictions available. Generate predictions first.")
        return

    # Calculate mean confidence for each class
    confidence_means = {cls: np.mean(confidences) for cls, confidences in class_confidences.items()}

    plt.figure(figsize=(14, 8))

    # Sort classes by count
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    # Limit to top 30 classes for readability
    if len(classes) > 30:
        classes = classes[:30]
        counts = counts[:30]
        plt.title('Top 30 Predicted Classes Distribution')
    else:
        plt.title('Predicted Classes Distribution')

    # Create bar chart
    bars = plt.bar(classes, counts, color='skyblue')

    # Add confidence labels above bars
    for i, (cls, count) in enumerate(zip(classes, counts)):
        if cls in confidence_means:
            conf = confidence_means[cls]
            plt.text(i, count + 0.5, f"{conf:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Predicted Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.figtext(0.91, 0.85, 'Mean Confidence', fontweight='bold')
    plt.tight_layout()

    # Save plot
    output_path = f'{config["outputs_dir"]}/class_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to {output_path}")
    plt.show()

