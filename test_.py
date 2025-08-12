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
from chromadb.api.types import EmbeddingFunction  # Import EmbeddingFunction base class

# Load environment variables
load_dotenv()
client = genai.Client()
# For local persistence:
# import chromadb.PersistentClient
# chromadb_client = chromadb.PersistentClient(path="./chroma_data")

# Your existing Gemini API request code...
def gemini_api_request(prompt, image, timeout=25):
    def make_request():
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image, prompt]
            )
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

# Load embedding model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Define the proper EmbeddingFunction wrapper class
class MyEmbeddingFunction(EmbeddingFunction):
    def _init_(self, model):
        self.model = model

    def _call_(self, texts):
        # texts is a list of strings, returns list of embedding vectors
        return self.model.encode(texts).tolist()

embedding_fn = MyEmbeddingFunction(model)

# Image folder path
image_folder = "clothes_tryon_dataset/train/cloth"
max_images = 2

# Filter image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
image_files = image_files[:max_images]

start_time = time.time()

for idx, img_file in enumerate(image_files, 500):
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
    "details": "string"
    "season": "string"
    "Silhouette: "string"
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
    "neckline": "collared",`
    "fit": "regular",
    "material_texture": "cotton",
    "closure_type": "button-up",
    "details": "n/a"
    "season": "spring"
    "Silhouette": "{H, X,A,C, Y }"
    "hue" : "hue_value",
    "brightness" : "brightness_value",
    "saturation": "saturation_value",
    "temperature" : "temperature_value"
}
    """

    attributes_JSON = gemini_api_request(prompt, image)
    if attributes_JSON is None:
        continue
    
 

    match = re.search(r'json\n(.*?)\n', attributes_JSON, re.DOTALL)
    if match:
        attributes_JSON_str = match.group(1)
    else:
        attributes_JSON_str = attributes_JSON
    try:
        item_attributes = json.loads(attributes_JSON_str)
        if not isinstance(item_attributes, dict): # Ensure it's a dictionary
            raise ValueError("Parsed JSON is not a dictionary.")
        if not all(k in item_attributes for k in ["category", "clothing_type", "dominant_color"]):
            raise ValueError("Missing essential attributes in JSON.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing or validating JSON for {img_file}: {e}. Raw output: {attributes_raw_text[:200]}... Skipping.")
        continue
        

    summary_prompt = f"""
    You are a fashion assistant. Based on the following clothing attributes, write a short, natural-sounding summary describing the item. Avoid technical jargon. Use normal everyday language.

    Attributes:
    {attributes_JSON}

    Summary:
    """