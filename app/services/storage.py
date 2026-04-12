import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

class SupabaseStorage:
    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.bucket_name = "project-files"

    def upload_file(self, file_path: str, destination_path: str):
        with open(file_path, "rb") as f:
            response = self.client.storage.from_(self.bucket_name).upload(
                destination_path, f, {"content-type": "text/csv"}
            )
        return response

    def upload_json(self, destination_path: str, json_data: str):
        response = self.client.storage.from_(self.bucket_name).upload(
            destination_path, 
            json_data.encode("utf-8"), 
            {"content-type": "application/json"}
        )
        return response

    def get_public_url(self, file_path: str):
        return self.client.storage.from_(self.bucket_name).get_public_url(file_path)

    def download_file(self, file_path: str, local_path: str):
        with open(local_path, "wb") as f:
            res = self.client.storage.from_(self.bucket_name).download(file_path)
            f.write(res)
        return local_path

storage_service = SupabaseStorage()
