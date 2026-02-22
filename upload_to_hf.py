"""Upload Sakhi project to Hugging Face Spaces — without LFS."""
import os
import shutil
import tempfile

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from huggingface_hub import HfApi

api = HfApi()
repo_id = "Hush04/sakhi-api"
src = "/Users/abhigupta/Desktop/Sakhi-1"

# Create a clean temp copy WITHOUT .git and .gitattributes (removes LFS tracking)
tmp = tempfile.mkdtemp()
dst = os.path.join(tmp, "sakhi")
print("📦 Creating clean copy without Git LFS tracking...")

ignore = shutil.ignore_patterns(
    ".venv", "__pycache__", ".git", "*.pyc", ".DS_Store",
    "backend.log", "node_modules", "*.pkl", "*.jsonl",
    "upload_to_hf.py", ".gitattributes"
)
shutil.copytree(src, dst, ignore=ignore)

# Remove the .gitattributes from the copy so HF doesn't try LFS
gitattr = os.path.join(dst, ".gitattributes")
if os.path.exists(gitattr):
    os.remove(gitattr)

print("🚀 Uploading to Hugging Face Spaces...")
print("   This may take a few minutes...")

api.upload_folder(
    folder_path=dst,
    repo_id=repo_id,
    repo_type="space",
)

# Cleanup
shutil.rmtree(tmp)

print("✅ Upload complete! Your Space is building at:")
print(f"   https://huggingface.co/spaces/{repo_id}")
