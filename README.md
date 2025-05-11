# Face Re-Identification System

<!-- ![Face Recognition Visualization](notebooks/person_metrics_visualization.png) -->

A scalable system for face recognition and re-identification across different projects and persons.

## Features

- 🖼️ Face embedding extraction and storage using DeepFace
- 🔍 Similarity search across face embeddings using Qdrant
- 📁 Project-based organization of persons and images
- 🚀 FastAPI-based REST API
- 🐳 Docker-ready deployment
- 📊 Jupyter notebooks for analysis and visualization
- 💾 MongoDB integration for metadata storage

## Project Structure

```
├── docker/                  - Docker configuration
├── notebooks/               - Analysis notebooks
├── src/                     - Main application source
│   ├── assets/              - Storage for files and databases
│   ├── controllers/         - Business logic handlers
│   ├── helpers/             - Helper functions and utilities
│   ├── models/              - Data models and schemas
│   ├── routes/              - API endpoints
│   ├── stores/              - Vector database integration
│   └── main.py             - Application entry point
└── requirements.txt         - Python dependencies
```

## Technologies Used

- **Backend**: FastAPI 0.110.2
- **Vector Database**: Qdrant 1.10.1
- **Face Recognition**: DeepFace 0.0.93
- **Document Database**: MongoDB (via Motor 3.6.0)
- **Storage**: Local filesystem with MongoDB integration
- **DevOps**: Docker

## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker and Docker Compose

### Installation

1. Clone the repository:

   ```bash
   git clone -b fast-api-working100%25 https://github.com/InceptionISA/Face-ReIdentification.git
   cd Face-ReIdentification
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

Then edit the `.env` file to match your local setup. Make sure to set the following environment variables:

#### MongoDB Configuration
- `MONGO_INITDB_ROOT_USERNAME`
- `MONGO_INITDB_ROOT_PASSWORD`
- `MONGODB_URL`
- `MONGODB_DATABASE`

#### Application Configuration
- `APP_NAME`
- `APP_VERSION`
- `FILE_ALLOWED_TYPES`
- `FILE_MAX_SIZE`
- `FILE_DEFAULT_CHUNK_SIZE`

#### Vector Database Configuration
- `VECTOR_DB_BACKEND`
- `VECTOR_DB_PATH`
- `VECTOR_DB_DISTANCE_METHOD`

#### Face Recognition Configuration
- `FACE_EMBEDDING_BACKEND`
- `EMBEDDING_SIZE`
- `EMBEDDING_BATCH_SIZE`
- `FACE_DETECTION_BACKEND`
- `MTCNN_MIN_FACE_SIZE`

### Running with Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

This will start all required services including:
- MongoDB database (accessible on port 27008)
  - Data is persisted in a Docker volume named `mongodata_face`
  - Uses MongoDB 7.0 with Ubuntu Jammy base image
  - Configured with root username and password from environment variables

The services are connected through a Docker network named `backend_face`.

### Running Locally

Start the FastAPI server:
```bash
uvicorn src.main:app --reload
```

## API Documentation

After starting the server, access the interactive API docs at:

- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

### API Endpoints

#### Face Recognition Endpoints (`/api/v1/faces`)

- `POST /{project_id}/persons/{person_id}/generate-embeddings` - Generate face embeddings for a specific person
- `POST /{project_id}/generate-embeddings-batch` - Generate face embeddings for all persons in a project
- `POST /{project_id}/search-similar` - Search for similar faces in a project
- `GET /{project_id}/vector-db-info` - Get vector database information for a project

#### Data Management Endpoints (`/api/v1/data`)

- `POST /{project_id}/persons/{person_id}/upload-image` - Upload an image for a person
- `DELETE /{project_id}` - Delete an entire project and all its data
- `DELETE /{project_id}/persons/{person_id}` - Delete a person and their data from a project

### Usage Examples

#### Uploading Images

```bash
curl -X POST -F "file=@test.jpg" -F "name=John Doe" -F "age=25" http://localhost:8000/api/v1/data/1/youssef/upload-image
```

#### Generating Face Embeddings

```bash
# For a single person
curl -X POST http://localhost:8000/api/v1/faces/1/persons/youssef/generate-embeddings

# For all persons in a project
curl -X POST http://localhost:8000/api/v1/faces/1/generate-embeddings-batch
```

#### Searching Similar Faces

```bash
curl -X POST -F "file=@test.jpg" "http://localhost:8000/api/v1/faces/1/search-similar?limit=5"
```

#### Getting Vector DB Info

```bash
curl -X GET http://localhost:8000/api/v1/faces/1/vector-db-info
```

#### Deleting Data

```bash
# Delete a person
curl -X DELETE http://localhost:8000/api/v1/data/1/persons/youssef

# Delete a project
curl -X DELETE http://localhost:8000/api/v1/data/1
```

## Development

### Notebooks

Jupyter notebooks for analysis are in the `notebooks/` directory. Start Jupyter with:

```bash
jupyter notebook
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

