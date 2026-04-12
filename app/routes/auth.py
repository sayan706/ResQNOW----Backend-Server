from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from app.services.db.supabase import execute_query, fetch_row
from app.services.auth_utils import get_password_hash, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
async def register(user: UserRegister):
    hashed_password = get_password_hash(user.password)
    try:
        query = """
        INSERT INTO users (username, email, password_hash)
        VALUES ($1, $2, $3)
        RETURNING id;
        """
        row = await fetch_row(query, user.username, user.email, hashed_password)
        return {"message": "User registered successfully", "user_id": row["id"]}
    except Exception as e:
        if "unique constraint" in str(e).lower():
            raise HTTPException(status_code=400, detail="Username or email already exists")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login")
async def login(user: UserLogin):
    query = "SELECT id, password_hash FROM users WHERE email = $1;"
    db_user = await fetch_row(query, user.email)
    
    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token(data={"sub": str(db_user["id"])})
    return {"access_token": access_token, "token_type": "bearer"}
