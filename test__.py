import os
import time
import json
from PIL import Image
import re
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from chromadb.api.types import EmbeddingFunction
import google.generativeai  as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client = genai.GenerativeModel("gemini-2.0-flash")

# Initialize ChromaDB
chromadb_client = chromadb.Client()

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

class MyEmbeddingFunction(EmbeddingFunction):
    def _init_(self, model):
        self.model = model

    def _call_(self, texts):
        return self.model.encode(texts).tolist()

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list:
        return self.model.encode([text])[0].tolist()

embedding_fn = MyEmbeddingFunction(model)

# Delete old collection if exists
if "fashion_summaries" in [c.name for c in chromadb_client.list_collections()]:
    chromadb_client.delete_collection("fashion_summaries")

collection = chromadb_client.create_collection(
    name="fashion_summaries",
    embedding_function=embedding_fn
)

def gemini_api_request(prompt, image=None, timeout=25):
    def make_request():
        try:
            content = [image, prompt] if image else ([prompt] if not isinstance(prompt, list) else prompt)
            response = client.generate_content(content)
            return response.text
        except Exception as e:
            print(f"API Error: {e}")
            return None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(make_request)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print("Request timed out.")
            return None
done=120
# Image folder path
image_folder = "clothes_tryon_dataset/train/cloth"
max_images = 2000 # Change as needed
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
])[:max_images]

# Main processing
for idx, img_file in enumerate(image_files[done:]):
    img_path = os.path.join(image_folder, img_file)
    image = Image.open(img_path)

    prompt = """
You are an expert fashion analyst. Carefully examine the clothing item in the provided image.
Your task is to extract its key fashion attributes.
Pay close attention to details, especially variations in color and specific design elements.

Provide the extracted attributes as a JSON object. Ensure all string values are lowercase and replace any spaces with hyphens (e.g., 'long-sleeve', 'crew-neck'). If an attribute is not present or clearly identifiable, use 'n/a' as its value.

JSON Schema:
{
    "category": "string",
    "clothing_type": "string",
    "dominant_color": "string",
    "secondary_color": "string",
    "color_details": "string",
    "pattern": "string",
    "sleeve_type": "string",
    "neckline": "string",
    "fit": "string",
    "material_texture": "string",
    "closure_type": "string",
    "details": "string",
    "season": "string",
    "silhouette": "string",
    "hue": "string",
    "brightness": "string",
    "saturation": "string",
    "temperature": "string"
}

Example:
{
    "category": "top",
    "clothing_type": "shirt",
    "dominant_color": "blue",
    "secondary_color": "white",
    "color_details": "blue-with-white-collar",
    "pattern": "striped",
    "sleeve_type": "long-sleeve",
    "neckline": "collared",
    "fit": "regular",
    "material_texture": "cotton",
    "closure_type": "button-up",
    "details": "n/a",
    "season": "spring",
    "silhouette": "a",
    "hue": "cool",
    "brightness": "medium",
    "saturation": "high",
    "temperature": "warm"
}
"""

    attributes_JSON = gemini_api_request(prompt, image)
   
    if attributes_JSON is None:
        continue

    # Extract JSON string from code block or plain response
    match = re.search(r'json\n(.*?)\n', attributes_JSON, re.DOTALL)
    attributes_JSON_str = match.group(1) if match else attributes_JSON

    try:
        item_attributes = json.loads(attributes_JSON_str)
        if not isinstance(item_attributes, dict):
            raise ValueError("Parsed JSON is not a dictionary.")
        if not all(k in item_attributes for k in ["category", "clothing_type", "dominant_color"]):
            raise ValueError("Missing essential attributes.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f" JSON parsing error for {img_file}: {e}")
        print(f"Raw output: {attributes_JSON_str[:200]}...\nSkipping.")
        continue

    # Create summary prompt
    summary_prompt = f"""
You are a fashion assistant. Based on the following clothing attributes, write a short, natural-sounding summary describing the item. Avoid technical jargon. Use normal everyday language.

Attributes:
{json.dumps(item_attributes, indent=2)}

Summary:
"""

    summary = gemini_api_request(summary_prompt, image)
    print(f"Generated summary for {img_file}: {summary}")
    if summary is None:
        continue

    item_id = os.path.splitext(img_file)[0]
    metadata_to_store = {"image_file": img_file}
    metadata_to_store.update(item_attributes)

    collection.add(
        documents=[summary],
        metadatas=[metadata_to_store],
        ids=[item_id]
    )

    print(f"✅ Processed {img_file} ({idx}/{len(image_files)})")