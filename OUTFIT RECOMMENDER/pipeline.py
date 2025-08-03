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
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key="AIzaSyBfS455VIwG3dttvADiDuKwAgNALvIgJAQ")
client = genai.GenerativeModel("gemini-2.0-flash")

def gemini_api_request(prompt, image=None, timeout=25):
    def make_request():
        try:
            # Build the content list correctly
            if image is not None:
                content = [image, prompt]  # both are valid types (e.g., Image + string)
            else:
                content = [prompt]  # just prompt

            # Ensure no None values are passed
            content = [c for c in content if c is not None]

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


# Initialize ChromaDB
import chromadb

chromadb_client = chromadb.PersistentClient(path="./chroma_storage")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, texts):
        return self.model.encode(texts).tolist()

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list:
        return self.model.encode([text])[0].tolist()

embedding_fn = MyEmbeddingFunction(model)





collection = chromadb_client.get_or_create_collection(name="fashion_summaries2",
    embedding_function=embedding_fn)

vectorstore1 = Chroma(
    collection_name="fashion_summaries2",
    client=chromadb_client,
    embedding_function=embedding_fn,
)
done=1188
# Image folder path
image_folder = "clothes_tryon_dataset/train/cloth"
max_images = 2000 # Change as needed
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
])[:max_images]



from typing import Dict, Any

MOCK_USERS_DATA = [
    {
        "user_id": "user_1",
        "explicit_prefs": {
            "preferred_brands": ["Lululemon", "Zara", "Uniqlo"],
            "preferred_styles": ["casual chic", "athleisure", "bohemian"],
            "preferred_colors": ["soft navy", "cream", "dusty rose"],
            "preferred_materials": ["modal", "linen", "cotton blend"],
            "preferred_fit": ["slim fit", "relaxed fit"],
            "sizes": {"tops": "S", "bottoms": "28x30"},
            "color_tone": "neutral",
            "body_shape": "hourglass",
        },
        "implicit_prefs": {
            "last_purchase_category": "casual tops",
            "browsing_interest": ["summer dresses", "lightweight jackets", "loose blouses"],
            "avg_price_point": "mid-range",
            "recent_search_terms": ["linen wrap dress", "flowy tops with sleeves", "boho chic skirts"]
        },
        "demographics": {
            "age_range": "22-30",
            "gender": "female",
            "location": "urban"
        },
        "last_viewed_item": {
            "item_id": "item_12345",
            "viewed_at": "2023-10-01T12:00:00Z",
            "item_attributes": {
                "category": "dress",
                "clothing_type": "wrap dress",
                "dominant_color": "soft navy",
                "secondary_color": "cream",
                "color_details": "navy-with-cream-accents",
                "pattern": "solid",
                "sleeve_type": "short-sleeve",
                "neckline": "v-neck",
                "fit": "relaxed fit",
                "material_texture": "linen blend",
                "closure_type": "wrap-around",
                "details": "flowy hem, lightweight fabric",
                "season": "summer",
                "silhouette": "a-line"
            }
        }
    }
]



def generate_user_profile_summary() -> str:
    user_profile = MOCK_USERS_DATA[0]

    prompt = f"""
You are an AI assistant specializing in personalized fashion recommendations.

Given the following detailed structured user profile data, generate a clear, concise, and natural-sounding summary that captures the user's fashion identity.

Focus on:
- Explicit preferences such as favorite brands, preferred styles, colors, materials, and fit.
- Implicit preferences like recent browsing interests, last purchases, and typical price range.
- Body type and how it influences style choices.
- Any notable recent search terms or fashion trends the user is engaging with.

Produce a 2-4 sentence summary that seamlessly integrates these aspects to convey a vivid sense of the user’s unique fashion tastes and shopping behavior.

Example:
“A young urban woman with a preference for casual chic and athleisure styles, favoring soft navy and dusty rose colors in lightweight, breathable fabrics like linen and modal. She tends to choose relaxed and slim fits from brands like Lululemon and Zara, often browsing summer dresses and flowy tops, reflecting a mid-range price point and an hourglass body type.”

Here is the user data:
{user_profile}
"""

    summary = gemini_api_request(prompt)
    if not summary:
        return "Could not generate summary for user."

    return summary

# Usage
import os

# keys for the services we will use

os.environ["GOOGLE_API_KEY"] = "AIzaSyBfS455VIwG3dttvADiDuKwAgNALvIgJAQ"

    

import uuid
from base64 import b64decode
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from sentence_transformers import SentenceTransformer
from langchain.retrievers.multi_vector import MultiVectorRetriever
from chromadb.api.types import EmbeddingFunction

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# Embedding function wrapper
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    top_description = kwargs.get("user_query", "No description provided.")  # Add this line
    summary_text = kwargs.get("user_preferences", "None provided.")

    context_text = ""
    for text_element in docs_by_type["texts"]:
        context_text += str(text_element) + "\n"

    prompt_template = f"""
You are an intelligent fashion assistant. Use only the information provided in the following context, which may include text, tables, or images. Additionally, consider the user's preferences when crafting your recommendation. Do not rely on external knowledge or assumptions.

Your goal is to recommend detailed complementary pieces — specifically lowers and accessories — that perfectly align with the user's preferences, event, and setting, based on the provided top description or image.

Describe the lowers and accessories with rich detail about fabric, texture, cut, color, style, and mood, ensuring they complement the given top.

=== Context ===
{context_text}

=== User Preferences ===
{summary_text}

=== Top Description / Image Description ===
{top_description}

Based on the above, provide a detailed recommendation for lowers and accessories that:
- Match the user's body type, season, occasion, and aesthetic.
- Complement the fabric, texture, color, and style of the described top.
- Evoke the tone the user wants (e.g., bold, soft, elegant, casual).
- Include details like fabric type, cut, color palette, and design elements for both lowers and accessories.

Example output:
"A pair of high-waisted, tailored white linen trousers with a relaxed fit, perfect for summer warmth and breezy comfort. The fabric has a subtle natural texture that complements the smooth silk top. For accessories, a tan leather belt with a minimalist gold buckle enhances the waistline. Pair this with delicate gold hoop earrings and a straw woven tote bag for a chic, effortless look ideal for a daytime formal event."

"""

    prompt_content = [{"type": "text", "text": prompt_template}]
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })
    
    return ChatPromptTemplate.from_messages([
        HumanMessage(content=prompt_content)
    ])


def generate_fashion_recommendation(
    user_query: str ,
    user_preferences: str = "None provided.",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    rag_collection_name: str = "multi_modal_rag"
) -> dict:
    """
    Runs the full RAG pipeline to generate a fashion top recommendation based on user query and preferences.

    Args:
        user_query: The user's natural language question.
        user_preferences: Optional string summary of user preferences.
        embedding_model_name: Sentence transformer model name to load embeddings.
        rag_collection_name: The Chroma collection name to use for retrieval.

    Returns:
        Dict with keys:
          - 'response': The Gemini-generated fashion recommendation text.
          - 'context_texts': List of texts retrieved and used in prompt.
    """

    # Load embedding model and create embedding function
    model = SentenceTransformer(embedding_model_name)
    embedding_fn = MyEmbeddingFunction(model)

    # Setup vectorstore retriever
    vectorstore = Chroma(collection_name=rag_collection_name, embedding_function=embedding_fn)
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Setup Gemini LLM
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Build the chain pipeline
    chain = (
        {
    "context": lambda x: parse_docs(retriever.invoke(x["user_query"])),
    "user_query": lambda x: x["user_query"],
    "user_preferences": lambda x: x["user_preferences"],
}

        | RunnableLambda(build_prompt)
        | gemini_llm
        | StrOutputParser()
    )

    # Prepare inputs and run

    inputs = {
    "user_query": user_query,
    "user_preferences": user_preferences
}
    context_texts = retriever.invoke(inputs["user_query"])
    response = chain.invoke(inputs)
    return {
    "response": response,
    "context_texts": context_texts,
}



# Example usage:
if __name__ == "__main__":
    user_query = "This is a fun and flashy sequined slip dress with a V-neck. It's sleeveless, has a slim fit, and features a black and gold ombre effect. Perfect for any season!"
    user_prefs = "A young urban woman with a preference for casual chic and athleisure styles, favoring soft navy and dusty rose colors."
    print("user_query:", user_query)
    print("user_preferences:", user_prefs)


    result = generate_fashion_recommendation(user_query, user_prefs)
    print("Fashion Recommendation:\n", result["response"])
    print("\nContext texts used:")
    for ctx in result["context_texts"]:
        print("-", ctx)


# Example usage:


# json_path = 'vitonhd_train_tagged.json'
# with open(json_path, 'r', encoding='utf-8') as f:
#     products = json.load(f)
# from langchain.document_loaders import JSONLoader




# loader = JSONLoader(
#     file_path='vitonhd_train_tagged.json',  
#     jq_schema = '.data[]',    
#     text_content=False,
#     json_lines=False
# )

# fashion_data = loader.load()


# print(f"Loaded {len(fashion_data)} products")

# # Each item is a Document with page_content as the JSON dict
# first_product = fashion_data[0].page_content

# print(first_product)
# from langchain.docstore.document import Document
# import json

# # Assume you loaded your JSON like this:
# with open('vitonhd_train_tagged.json', 'r', encoding='utf-8') as f:
#     fashion_data = json.load(f)

# fashion_docs_processed = []

# # Loop over the list inside the "data" key
# for product in fashion_data["data"]:
#     metadata = {
#         "title": product.get("file_name", "unknown"),
#         "id": "fashion-dataset",
#         "source": "dataset",
#         "category": product.get("category_name", "unknown"),
#         "page": 1
#     }

#     tags = [
#         f"{tag['tag_name']}: {tag['tag_category']}"
#         for tag in product.get("tag_info", [])
#         if tag['tag_category'] is not None
#     ]

#     images_info = [
#         f"Image {i+1} - Path: {img['image_path']}, Size: {img['image_width']}x{img['image_height']}"
#         for i, img in enumerate(product.get("image", []))
#     ]

#     data = ' '.join([
#         f"Category: {product.get('category_name', '')}",
#         "Tags: " + ', '.join(tags),
#         "Images: " + ' | '.join(images_info)
#     ])

#     fashion_docs_processed.append(Document(page_content=data, metadata=metadata))

# print(f"Processed {len(fashion_docs_processed)} documents.")
# print(fashion_docs_processed[0].page_content)  # preview first doc content


from langchain.retrievers import BM25Retriever, EnsembleRetriever

# BM25 Retriever
# bm25_retriever = BM25Retriever.from_documents(
#     documents=fashion_docs_processed,  # Make sure total_docs is a list of LangChain Documents
#     k=5
# )

# Chroma Similarity Retriever
similarity_retriever = vectorstore1.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

import re
import json
from enum import Enum
from typing import List, Dict, Any

# Assuming these imports are done globally:
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
# ... plus other imports from your environment


class RecommendationIntent(Enum):
    GENERAL_TOP = "GENERAL_TOP_RECOMMENDATION"
    TOP_FOR_LOWER = "TOP_FOR_LOWER_RECOMMENDATION"
    SIMILAR_IMAGE_TOP = "SIMILAR_IMAGE_TOP_RECOMMENDATION"


def classify_intent(original_query: str, has_image: bool = False) -> RecommendationIntent:
    query_lower = original_query.lower()
    if has_image and ("similar to this" in query_lower or "image of" in query_lower or "similar shirt" in query_lower):
        return RecommendationIntent.SIMILAR_IMAGE_TOP
    elif any(x in query_lower for x in ["top for", "goes with", "matching top", "pair with"]):
        return RecommendationIntent.TOP_FOR_LOWER
    else:
        return RecommendationIntent.GENERAL_TOP

genai.configure(api_key="AIzaSyBfS455VIwG3dttvADiDuKwAgNALvIgJAQ")
client = genai.GenerativeModel("gemini-2.0-flash")

def gemini_api_request(prompt, image=None, timeout=25):
    def make_request():
        try:
            # Build the content list correctly
            if image is not None:
                content = [image, prompt]  # both are valid types (e.g., Image + string)
            else:
                content = [prompt]  # just prompt

            # Ensure no None values are passed
            content = [c for c in content if c is not None]

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
def is_minimal_direct_query(query: str) -> bool:
    """
    Detects if the query is short and direct, e.g., 'red top with floral',
    meaning it should skip profile/rule expansion.
    """
    keywords = query.lower().split()
    return len(keywords) <= 6 and not any(
        qword in query.lower() for qword in ["wearing", "pair", "go with", "match", "occasion"]
    )


def generate_llm_query(
    intent: RecommendationIntent,
    original_query: str,
    user_profile: str,
    applicable_rules_text: str,
    image_description: str = None
) -> Dict[str, Any]:
    full_llm_input = f"""
User Query: {original_query}

**User Profile:**
{user_profile}

**Applicable Rules (Retrieved from KB):**
{applicable_rules_text}
"""
    if is_minimal_direct_query(original_query):
        # Minimal direct query — skip profile/rule expansion
        prompt = f"""
    User Query: {original_query}

    Generate a short and precise structured query for semantic search.
    Do NOT add extra context from user profile or rules.
    Focus on keeping the attributes directly visible in the query, like color, pattern, item_type.

    When generating the structured query for the vector database, please note:
- The query must explicitly be for 'top' items.
- Describe the top's style, color, pattern, neckline, sleeve type, and material.
- If the user query or user preferences specify one or more colors, those colors should be treated as highly important and prioritized in the query, even if they conflict with other preferences or rules.
- Ensure the color(s) from the query or user preferences appear prominently in the JSON output.

Output format should be a JSON object with keys like 'item_type', 'style', 'color', 'pattern', 'neckline', 'sleeve_type', 'fabric', 'keywords'.


    Output format:
    {{"item_type": [...], "color": [...], "pattern": [...], "keywords": [...] }}

    """
    else:
        # Normal behavior — use user_profile + applicable_rules
        if intent == RecommendationIntent.GENERAL_TOP:
            prompt = f"""
    {full_llm_input}
    Based on the 'User Query', 'User Preferences', and 'Applicable Rules', generate a precise structured query for a vector database to retrieve **top clothing items**. Focus on keywords and attributes for styles, colors, occasions, fit, fabric, and seasonality.
Output format should be a JSON object with keys like 'item_type', 'style', 'color', 'fabric', 'season', 'keywords'. Values should be lists of strings or single strings.
Example: {{"item_type": ["top"], "style": ["casual"], "color": ["blue"], "keywords": ["comfortable"]}}

for example : on the basis of rules , for a query like "I have a formal event coming up and need an elegant, polished outfit. Can you recommend something appropriate and fashionable for the occasion?"
you should be able to distinguish the fact and give her shirts , formal tops . 
for example : 
I'm packing for a beach vacation and want some outfit ideas that are breezy, stylish, and perfect for warm weather. Any recommendations?


When generating the structured query for the vector database, please note:
- The query must explicitly be for 'top' items.
- Describe the top's style, color, pattern, neckline, sleeve type, and material.
- If the user query or user preferences specify one or more colors, those colors should be treated as highly important and prioritized in the query, even if they conflict with other preferences or rules.
- Ensure the color(s) from the query or user preferences appear prominently in the JSON output.

Output format should be a JSON object with keys like 'item_type', 'style', 'color', 'pattern', 'neckline', 'sleeve_type', 'fabric', 'keywords'.

Example:
{{"item_type": ["top"], "neckline": ["crew-neck"], "color": ["blue"], "pattern": ["striped"], "fabric": ["cotton"]}}
"""
    

        elif intent == RecommendationIntent.TOP_FOR_LOWER:
                lower_match = re.search(r'(?:top for|goes with|matching top for|pair with)\s+(.*?)(?:\?|$)', original_query.lower())
                lower_description = lower_match.group(1).strip() if lower_match else "unspecified lower"

                prompt = f"""
        {full_llm_input}
        The user has a lower clothing item in mind: '{lower_description}'.
        Based on the 'User Query', 'User Preferences', and 'Applicable Rules', generate a precise structured query for a vector database. The query should exclusively target **top garments** that would pair well with '{lower_description}', detailing attributes like style, color, fabric, and overall aesthetic compatibility.
        Output format should be a JSON object with keys like 'item_type', 'style', 'color', 'fabric', 'keywords', 'compatible_with_lower'. Values should be lists of strings or single strings.
        Example: {{"item_type": ["top"], "style": ["casual", "smart-casual"], "color": ["white", "blue"], "compatible_with_lower": ["dark-wash-skinny-jeans"]}}
        """
        elif intent == RecommendationIntent.SIMILAR_IMAGE_TOP:
            prompt = f"""
    {full_llm_input}
    The user has provided an image of a top described as: '{image_description}'.
    Based on the image description, focus more on image description, generate a comprehensive structured query for a vector database. The query should describe the top's style, color, pattern, neckline, sleeve type, and material. Ensure the query is explicitly for 'top' items.
    Focus on generating a query that captures the essence of the top in the image, including its style, color, and any notable features. mainly  focus more on color_details and clothing_type.
    """
    llm_response_text = gemini_api_request(prompt)

    if llm_response_text:
        match = re.search(r'```json\n(.*?)\n```', llm_response_text, re.DOTALL)
        json_str = match.group(1) if match else llm_response_text

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Error parsing JSON from LLM response. Returning fallback query.")
            return {"item_type": ["top"], "keywords": ["general", "clothing"]}
    else:
        print("LLM did not return a response for query generation.")
        return {"item_type": ["top"], "keywords": ["general", "clothing"]}


def your_rag_pipeline(
    query: str,
    user_profile_summary: str,
    rules_text: str,
    similarity_retriever,
    reranker
) -> List[Dict[str, str]]:
    """
    Final pipeline function:
    - Classify intent
    - Generate vector DB query via LLM
    - Retrieve documents from ensemble retriever
    - Rerank with cross-encoder
    - Return top 5 recommendations with images
    """

    # Step 1: Classify user intent
    intent = classify_intent(query)

    # Step 2: Generate structured query for retrieval
    structured_query = generate_llm_query(
        intent=intent,
        original_query=query,
        user_profile=user_profile_summary,
        applicable_rules_text=rules_text
    )
    print(user_profile_summary)
    print(rules_text)
    # Step 3: Create string from structured query
    query_keywords = []
    for v in structured_query.values():
        if isinstance(v, list):
            query_keywords.extend(v)
        elif isinstance(v, str):
            query_keywords.append(v)
    query_string = " ".join(query_keywords)

    # Step 4: Retrieve documents with ensemble retriever
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[bm25_retriever, similarity_retriever],
    #     weights=[0, 1]
    # )
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=CrossEncoderReranker(model=reranker, top_n=5),
    #     base_retriever=ensemble_retriever
    # )
    retrieved_docs = similarity_retriever.get_relevant_documents(query)
    for doc in retrieved_docs:
        print(doc.metadata)


    # Step 5: Return top 5 documents as recommendations
    recommendations = []
    for doc in retrieved_docs[:5]:
        recommendations.append({
            "title": doc.metadata.get("title", "Top"),
            "description": doc.page_content.strip(),
            "image_file": f"http://localhost:8000/images/{doc.metadata.get('image_file', '')}"

        })



    return recommendations

def get_image_path(rec: dict) -> str:
    image_file = rec.get("image_file", "") or rec.get("title", "")
    if image_file:
        return os.path.join("clothes_tryon_dataset/train/cloth", image_file)
    return ""
from PIL import Image
from IPython.display import display
import os

def display_recommendation_with_image(recommendations: list):
    for rec in recommendations:
        print(f"Title: {rec.get('title', 'Untitled')}")
        print(f"Description: {rec.get('description', 'No description')}\n")

        image_path = get_image_path(rec)

        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                display(img)
            except Exception as e:
                print(f"[Image could not be displayed: {e}]")
        else:
            print("[No image available]")

        print("-" * 50)
        
reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")



from typing import Union
from PIL import Image

def find_similar_tops_from_image(
    user_image: Union[str, Image.Image],
    similarity_retriever,
) -> List[Dict[str, str]]:
    """
    Given an image input, find similar top clothing items from dataset.

    Args:
        user_image: Path to image file or PIL Image object.
        user_profile_summary: User profile text for personalization.
        rules_text: Applicable rules from knowledge base.
        similarity_retriever: The retriever to use for document retrieval.

    Returns:
        List of dicts with recommendation metadata.
    """

    # Step 1: Get a textual description from the image
    # This is a placeholder: replace with your actual image feature extraction or captioning logic
    # For now, assume a function `describe_image(image)` returns a descriptive string
    if isinstance(user_image, str):
        from PIL import Image
        image_obj = Image.open(user_image)
    else:
        image_obj = user_image
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
    "brand": "string",
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
{g
    "category": "top",
    "clothing_type": "shirt",
    "dominant_color": "blue",
    "secondary_color": "white",
    "color_details": "blue-with-white-collar",
    "brand: "vans",
    "pattern": "striped",
    "sleeve_type": "long-sleeve",
    "neckline": "collared",
    "fit": "regular",
    "material_texture": "cotton",
    "closure_type": "button-up",
    "details": "vans",
    "season": "spring",
    "silhouette": "a",
    "hue": "cool",
     "brightness": "medium",
    "saturation": "high",
    "temperature": "warm"
}
"""
    image_description = gemini_api_request(prompt=prompt, image=image_obj)
    print(f"Image Description: {image_description}")

    summary_prompt = f"""
You are a fashion assistant. Based on the following clothing attributes, write a short, natural-sounding summary describing the item. Avoid technical jargon. Use normal everyday language.

Attributes:
{json.dumps(image_description, indent=2)}

Summary:
"""
    summary = gemini_api_request(summary_prompt, image=None)
    print(summary)
    # Step 5: Retrieve documents based on query string
    retrieved_docs = similarity_retriever.get_relevant_documents(summary)

    # Step 6: Prepare and return top 5 recommendations
    recommendations = []
    for doc in retrieved_docs[:5]:
        recommendations.append({
            "title": doc.metadata.get("title", "Untitled"),
            "description": doc.page_content.strip(),
            "image_file": doc.metadata.get("image_file", "")
        })

    return recommendations

