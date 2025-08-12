from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

# Your pipeline imports here
from pipeline import (
    generate_user_profile_summary,
    generate_fashion_recommendation,
    your_rag_pipeline,
    similarity_retriever,
    reranker,
    find_similar_tops_from_image
)

app = FastAPI()

# Mount static folder for serving images
app.mount(
    "/images",
    StaticFiles(directory="static/clothes_tryon_dataset/train/cloth"),
    name="images"
)

# Templates folder
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model for JSON query
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_fashion(request: QueryRequest):
    query = request.query

    # Generate user profile and fashion rules
    user_profile_summary = generate_user_profile_summary()
    rules_text = generate_fashion_recommendation(query, user_profile_summary)

    # Call RAG pipeline for recommendations
    result = your_rag_pipeline(
        query=query,
        user_profile_summary=user_profile_summary,
        rules_text=rules_text,
        similarity_retriever=similarity_retriever,
        reranker=reranker
    )

    # Append full URL for image files in results
    for rec in result:
        if rec.get("image_file"):
        # Only prepend base URL if it's NOT already a full URL
            if not rec["image_file"].startswith("http"):
                rec["image_file"] = f"http://localhost:8000/images/{rec['image_file']}"
                print("Updated image URL:", rec["image_file"])
    return {"result": result}


@app.post("/image-query")
async def image_query(file: UploadFile = File(...)):
    try:
        # Save the uploaded image temporarily
        ext = file.filename.split('.')[-1]
        temp_image_path = f"temp_uploaded_image.{ext}"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("Uploaded file name:", file.filename)
        print("Saved as:", temp_image_path)
        print("File extension:", ext)
        print(temp_image_path)

        # Generate user profile and fashion rules
        
        # Call the new function (no image here, just text query)
        result = find_similar_tops_from_image(
            user_image=temp_image_path,
            similarity_retriever=similarity_retriever,
        )


        # Remove temp image
        os.remove(temp_image_path)

        # Add full URL for images
        for rec in result:
            if rec.get("image_file"):
                if not rec["image_file"].startswith("http"):
                    rec["image_file"] = f"http://localhost:8000/images/{rec['image_file']}"
                    print("Updated image URL:", rec["image_file"])
        final_output = {"recommendations": result}


        print("Image recommendations:", final_output)
        return final_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
from fastapi import Query

@app.get("/recommendations")
async def get_fashion_recommendation(top_description: str = Query(..., alias="top_desc")):
    user_prefs = "A young urban woman with a preference for casual chic and athleisure styles, favoring soft navy and dusty rose colors."
    result = generate_fashion_recommendation(top_description, user_prefs)
    return {
        "recommendations": [
            {
                "title": "Context Based Fashion Recommendation",
                "description": result["response"],
            }
        ]
    }
