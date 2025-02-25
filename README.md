# Face Re-Identification Repository

This repository implements the face re-identification component for the AI for FineTech Competition. It focuses on developing a system capable of matching images of staff faces to their corresponding identities, as well as identifying non-staff faces. The generated output file will later be merged with the tracking results in the Integration Repository.

## Overview

- **Objective:**  
  Develop a scalable face re-identification system that measures similarity between face images to accurately verify staff identities.
  
- **Evaluation Metric:**  
  Accuracy Score (Calculated as: Correct Identifications / Total Identifications)
  
- **Output:**  
  A CSV file (e.g., `outputs/face_reid_results{last_number}.csv`) containing face identification results formatted for integration.

## Directory Structure

```
face_reid_repo/
├── data/               # dataset, Scripts and notebooks for face image loading and preprocessing
├── models/             # Model weights, training scripts, and evaluation code for face re-identification
├── notebooks/          # Jupyter notebooks for experimentation and analysis
├── outputs/            # Folder to store output files (e.g., outputs/face_reid_results.csv)
├── utils/              # Utility functions (e.g., face alignment, similarity computation)
├── README.md           
└── requirements.txt   
```

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/face_reid_repo.git
   cd face_reid_repo
   ```

2. **Create a Virtual Environment and Install Dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Data Preparation:**

   - Ensure you have access to the required face image dataset.
   - Update data paths in the scripts or notebooks as needed.
   - Run any preprocessing scripts located in the `data/` folder to prepare the data.

## Usage

1. **Training and Evaluation:**

   - Train your face re-identification model using the scripts in the `models/` directory.
   - Evaluate the model's performance using the provided evaluation scripts.

   Example command:
   ```bash
   python models/train_face_reid.py --config config.yaml
   ```

2. **Generating Output:**

   - After training and evaluation, generate the final output file for face re-identification.
   - Save the output CSV file in the `outputs/` folder with the name `face_reid_results{last_number}.csv`.

   Example command:
   ```bash
   python models/generate_face_reid_output.py --output outputs/face_reid_results.csv
   ```

   **Output Format:**
   - **Columns:** `ID`, `Frame`, `Objects`, `Objective`
   - **Objects Column:** A dictionary containing keys such as `gt` (ground truth label) and `image` (path to the face image file).

## Integration with the Integration Repository

The final submission for the competition will be assembled in the Integration Repository. Your output file (`outputs/face_reid_results{last_number}.csv`) should adhere to the following:
- **Format:** Must match the sample provided in the Integration Repository’s README.
- **Purpose:** It will be merged with the tracking output to create the final submission file.

Refer to the [Integration Repository README](https://github.com/InceptionISA/Integration) for more details on the integration process.

## Experiment Tracking and Logging

- **Logging:** Record each experiment's hyperparameters, evaluation metrics, and observations in dedicated log files or within commit messages.
- **Versioning:** Use Git branches and descriptive commit messages to track progress and maintain reproducibility.
- **Documentation:** Utilize Jupyter notebooks or markdown files in the `notebooks/` directory to document experimental results and insights.

## Contributing

- Follow the repository guidelines and branch naming conventions.
- Submit pull requests for new features, improvements, or bug fixes.
- Ensure thorough testing before merging changes into the main branch.


