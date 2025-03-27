# Face Re-Identification System

<!-- ![Face Recognition Visualization](notebooks/person_metrics_visualization.png) -->

A scalable system for face recognition and re-identification across different projects and persons.

## Features

- 🖼️ Face embedding extraction and storage
- 🔍 Similarity search across face embeddings
- 📁 Project-based organization of persons and images
- 🚀 FastAPI-based REST API
- 🐳 Docker-ready deployment
- 📊 Jupyter notebooks for analysis and visualization

## Project Structure

```
├── docker/                  - Docker configuration
├── notebooks/               - Analysis notebooks
├── src/                     - Main application source
│   ├── assets/              - Storage for files and databases
│   ├── controllers/         - Business logic handlers
│   ├── models/              - Data models and schemas
│   ├── routes/              - API endpoints
│   ├── stores/              - Vector database integration
│   └── utils/               - Utility functions
├── inference.py             - Inference script
├── download_results.py      - Results download utility
└── requirements.txt         - Python dependencies
```

## Technologies Used

- **Backend**: FastAPI
- **Vector Database**: Qdrant
- **Face Recognition**: Deep learning models
- **Storage**: Local filesystem (with MongoDB-like organization)
- **DevOps**: Docker

## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker (for containerized deployment)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/face-reidentification.git
   cd face-reidentification
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

### Running with Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Running Locally

Start the FastAPI server:

```bash
uvicorn src.main:app --reload
```

## API Documentation

After starting the server, access the interactive API docs at:

- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

### Key Endpoints

- `POST /upload/{project_id}/{person_id}` - Upload face images
- `GET /search/{project_id}` - Search for similar faces
- `GET /projects` - List all projects
- `GET /persons/{project_id}` - List persons in a project

## Usage Examples

### Uploading Images

```bash
curl -X POST -F "file=@test.jpg" http://localhost:8000/upload/1/youssef
```

### Searching Similar Faces

```bash
curl -X GET "http://localhost:8000/search/1?limit=5"
```

## Development

### Running Tests

Tests can be run with:

```bash
pytest tests/
```

### Notebooks

Jupyter notebooks for analysis are in the `notebooks/` directory. Start Jupyter with:

```bash
jupyter notebook
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Qdrant team for the vector database
- FastAPI for the excellent web framework
- All open-source face recognition model contributors

```

This README includes:
1. Project overview and features
2. Clear directory structure explanation
3. Setup and installation instructions
4. API documentation
5. Usage examples
6. Development information
7. License and acknowledgments

You may want to customize:
- The license type if you're using something other than MIT
- Specific model credits if you're using particular face recognition models
- Additional deployment instructions if you have special requirements
- Team information if this is a collaborative project

Would you like me to add any specific sections or modify any part of this README?
```
