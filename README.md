# ResQNow Backend API

AI-based ambulance optimization and real-time prediction system.

## Tech Stack
- **FastAPI**: Async web framework
- **Supabase (PostgreSQL)**: Primary database
- **Supabase Storage**: CSV and JSON storage
- **Celery + Redis**: Background processing for LSCP optimization
- **Pulp**: Linear programming solver for ambulance placement
- **Google Gemini AI**: Intelligent risk reasoning

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- Redis Server (local or cloud)
- Supabase Project

### 2. Environment Configuration
Create a `.env` file in the root directory (see `.env.template` if any):
```env
DB_HOST=your_host
DB_PORT=6543
DB_NAME=postgres
DB_USER=your_user
DB_PASSWORD=your_password

SUPABASE_URL=your_project_url
SUPABASE_SERVICE_ROLE_KEY=your_key

JWT_SECRET=secure_random_string
REDIS_URL=redis://localhost:6379/0

GOOGLE_MAPS_API_KEY=your_key
GEMINI_API_KEY=your_key
```

### 3. Database Setup
Run the SQL script provided in `setup_supabase.sql` in your Supabase SQL Editor to create the necessary tables.

### 4. Installation
```bash
pip install -r requirements.txt
```

### 5. Running the Application

**Run the FastAPI server:**
```bash
uvicorn app.main:app --reload
```

**Run the Celery worker (in a separate terminal):**
```bash
celery -A worker.celery_app worker --loglevel=info
```

## API Usage
- **Auth**: Register and Login to get a JWT token.
- **Upload**: Upload a CSV with accident data. This triggers background processing.
- **Status**: Check if the project processing is complete and get the latest version.
- **Predict**: Provide coordinates to get real-time risk level and nearest ambulance assignment.

Testing: Use the included `postman_collection.json`.
