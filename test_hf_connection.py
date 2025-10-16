# test_hf_connection.py
from huggingface_hub import snapshot_download

print("🔹 Checking Hugging Face authentication...")

# Download model, ignoring deep TensorBoard logs to avoid Windows long path errors
model_path = snapshot_download(
    repo_id="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    ignore_patterns=["runs/*"]  # <-- skips unnecessary event logs
)

print("Model downloaded at:", model_path)
