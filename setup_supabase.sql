-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_url TEXT NOT NULL,
    status TEXT DEFAULT 'uploaded', -- uploaded, processing, completed, failed
    task_id TEXT, -- Celery task ID
    latest_state_id UUID, -- References project_states(id) - set after state is created
    parameters JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, file_hash)
);

-- Project States table
CREATE TABLE IF NOT EXISTS project_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    json_url TEXT NOT NULL,
    state_type TEXT NOT NULL, -- initial, updated
    version INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add foreign key constraint to projects for latest_state_id once project_states is defined
ALTER TABLE projects 
ADD CONSTRAINT fk_latest_state 
FOREIGN KEY (latest_state_id) 
REFERENCES project_states(id) 
ON DELETE SET NULL;

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_file_hash ON projects(file_hash);
CREATE INDEX IF NOT EXISTS idx_project_states_project_id ON project_states(project_id);
CREATE INDEX IF NOT EXISTS idx_project_states_version ON project_states(version DESC);
