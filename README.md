# How to Reproduce Face Re-Identification Results

This guide will help you reproduce the face re-identification results using our implementation. The system uses Facenet512 for face recognition and can identify and match faces across different images.

## Quick Start

1. **Install Dependencies**

```bash
pip install deepface numpy pandas matplotlib scikit-learn tqdm
```

2. **Prepare Dataset Structure**

```
dataset/
└── surveillance-for-retail-stores/
    └── face_identification/
        └── face_identification/
            ├── test/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            └── train/
                ├── person_0/
                ├── person_1/
                └── ...
```

3. **Run the Pipeline**

```python
from src.face_recognition import FaceRecognition

# Initialize with best performing configuration
config = {
    'model_name': 'Facenet512',
    'backend': 'mtcnn',  # Best performing face detector
    'similarity_threshold': 0.65,
}

face_rec = FaceRecognition(config)
face_rec.run_pipeline(
    train_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/train',
    test_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/test'
)
```

## Detailed Steps

### 1. Environment Setup

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install deepface numpy pandas matplotlib scikit-learn tqdm
```

### 2. Dataset Preparation

1. Create the following directory structure:

   ```
   dataset/
   └── surveillance-for-retail-stores/
       └── face_identification/
           └── face_identification/
               ├── test/          # Place test images here
               └── train/         # Place training images here
   ```

2. Training images should be organized in subdirectories:
   - Each person should have their own directory named `person_X` (e.g., `person_0`, `person_1`)
   - Place all images of a person in their respective directory

### 3. Running the Pipeline

Create a Python script (e.g., `run_pipeline.py`) with the following content:

```python
from src.face_recognition import FaceRecognition

# Best performing configuration
config = {
    'model_name': 'Facenet512',
    'backend': 'mtcnn',           # Best performing face detector
    'similarity_threshold': 0.65, # Optimized threshold
    'use_alignment': True,        # Enable face alignment
    'embeddings_dir': 'embeddings',
    'metrics_dir': 'metrics',
    'outputs_dir': 'outputs'
}

# Initialize the face recognition system
face_rec = FaceRecognition(config)

# Run the complete pipeline
face_rec.run_pipeline(
    train_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/train',
    test_dir='dataset/surveillance-for-retail-stores/face_identification/face_identification/test'
)
```

Run the script:

```bash
python run_pipeline.py
```

### 4. Expected Output

The pipeline will generate:

- Face embeddings in `embeddings/Facenet512/`
- Submission file in the root directory
- Statistics and visualizations in `metrics/`

## Configuration Options

| Parameter            | Description                 | Recommended Value |
| -------------------- | --------------------------- | ----------------- |
| model_name           | Face recognition model      | 'Facenet512'      |
| backend              | Face detection backend      | 'mtcnn'           |
| similarity_threshold | Threshold for face matching | 0.65              |
| use_alignment        | Enable face alignment       | True              |

## Notes

- The system uses Facenet512 as the best performing model
- MTCNN is the recommended face detector for best results
- The similarity threshold of 0.65 has been optimized for this dataset
- Results are reproducible with the same random seed (42)
