import hashlib

def generate_file_hash(file_content: bytes) -> str:
    """Generates SHA256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()
