# INTRODUCTION
This project presents an end-to-end AI-driven platform for personalized fashion recommendations and immersive virtual try-on experiences. 
Users can discover clothing items tailored to their style preferences and preview garments on their own images before purchase. 
The system combines cutting-edge computer vision, deep learning, and web technologies to deliver seamless integration of recommendation and try-on
functionalities within a responsive web interface.

# VIRTUAL TRY-ON

### 1) TPS Based Method
This model uses two key components: the Geometric Matching Module (GMM) and the Context-Aware Generator (CAG).
The GMM transforms target garments into warped versions for accurate spatial alignment, and the CAG synthesizes 
realistic try-on images using these warped clothes and various contextual elements. The entire pipeline is end-to-end
learnable, trained with L1 and perceptual losses for high fidelity and accuracy.

#### Geometric Matching Module
The purpose of this process is to transform the target garment into a warped version that aligns with the person’s pose. 
It begins by analyzing visual features from both the person and the clothing item to understand their structure and positioning.
Based on this analysis, the model calculates spatial transformation parameters that guide how the garment should be reshaped. 
Using these parameters, a Thin-Plate Spline (TPS) transformation is applied—this can be imagined as a flexible grid that 
bends and stretches to deform the garment image smoothly. The result is a naturally warped garment that fits the person’s body shape and pose.

<img width="1800" height="564" alt="Screenshot 2025-08-03 at 1 14 56 AM" src="https://github.com/user-attachments/assets/088d769b-4a6e-410e-ba4d-296393370625" />

#### Context Aware Generator
The purpose of this stage is to synthesize the final virtual try-on image by combining the warped garment with various contextual 
elements such as the person's body shape, hair, and background. The architecture employs ResNet blocks and upsampling layers to 
construct and refine the image. A key component of this process is the use of specialized Context-Aware Normalization (CAN) layers,
which incorporate contextual information at different resolutions to ensure semantic consistency and alignment throughout the image.
This hierarchical approach allows the model to progressively enhance and refine the visual details, ultimately generating a realistic
and contextually accurate virtual try-on result.

<img width="600" height="550" alt="Screenshot 2025-08-03 at 1 19 53 AM" src="https://github.com/user-attachments/assets/28848647-23fc-4b0d-b791-b945a1cd54ee" />

### 2) Diffusion-Based VITON with Stable Diffusion

This method extends the Stable Diffusion framework for virtual try-on by incorporating multiple input modalities to guide the 
image generation process. It uses a text prompt to describe garment attributes such as style, color, and patterns, alongside
spatially aligned visual inputs including a raw cloth image, a binary cloth mask, a person segmentation map, and a person-agnostic
image that represents the body shape and pose without clothing. These spatial inputs are concatenated along the channel dimension and 
passed through a 1x1 convolution layer before being fed into the UNet denoising model. This design enables early fusion of spatial 
conditioning inputs, preserving the core architecture of Stable Diffusion while enhancing its capability for spatial alignment and 
stylistic fidelity. As a result, the model can generate virtual try-on images where garments accurately follow body contours and
precisely reflect the specified styles, colors, and patterns.


### 3)  ControlNet-Enhanced Diffusion Try-On

This approach enhances structural consistency in virtual try-on by integrating ControlNet into the Stable Diffusion framework.
ControlNet introduces additional, trainable pathways that guide the generation process using extra spatial conditions—such as 
pose or edge maps—without the need to retrain the large, pre-trained diffusion model. It works by creating a duplicate of the 
encoder blocks from the diffusion model, which forms the ControlNet branches. Meanwhile, the original diffusion model remains 
frozen to preserve its learned capabilities. Each ControlNet branch is dedicated to a specific conditioning input and begins 
and ends with "zero convolution" layers—1x1 convolutions initialized to zero—to ensure that training starts without disrupting 
the base model’s output. The feature maps from these branches are injected into the corresponding layers of the frozen model 
through element-wise addition, allowing structural information to influence the denoising process. This setup successfully 
separates structural control (e.g., body pose and layout) from appearance generation (e.g., garment texture and style),
with structural maps providing spatial alignment and the text prompt defining stylistic attributes. The result is a highly 
controllable and robust virtual try-on system that performs well even under diverse poses and complex garments.


### 4) VITON-HD AND HR-VITON
In addition to experimenting with and developing our own custom VITON pipelines, we also implemented and tested publicly available pretrained virtual try-on models—specifically, VITON-HD and HR-VITON. These models were integrated into our system to enable rapid testing, comparison, and fallback support. We used the official model checkpoints and adapted the input-output interfaces to align with our framework. Including these baselines helped validate our approaches and offered alternative inference paths within our service.

## FASHION RECOMMENDATION SYSTEM

This intelligent fashion recommendation system blends visual understanding, user profiling, and contextual reasoning to deliver highly personalized fashion suggestions.

### 1. Core Recommendation Engine

#### 1.1 Data Processing & Feature Extraction
- Fashion items are enriched with detailed visual attributes using the **Gemini model**, including:
  - Category, clothing type, colors, patterns, sleeve type, neckline, fit, material texture, closure type, design details, season, silhouette, and color properties (hue, brightness, saturation, temperature).
- Semantic embeddings are generated with `sentence-transformers/all-MiniLM-L6-v2` for similarity search.

#### 1.2 User Profiling
- Builds a detailed user profile combining:
  - **Explicit preferences**: preferred brands, styles, colors, materials, fits, color tone, body shape.
  - **Implicit behaviors**: search terms, browsing history, previous purchases.
  - **Demographics**: age range, gender, location.
- Image inputs are processed with Gemini to extract visual context and refine preferences.

#### 1.3 Context-Aware Recommendation Engine (RAG-based)
- Uses **Retrieval-Augmented Generation (RAG)** for rule-informed recommendations.
- Retrieves from a curated knowledge base of fashion rules:
  - Color palette theory, body shape dressing guidelines, occasion-based dressing, seasonal fabrics.
- Combines **BM25** + **semantic search** and re-ranks with **CrossEncoder** for contextual accuracy.

#### 1.4 Query Classification & Prompt Templates
- Classifies user queries into:
  - **Search-like queries** (e.g., "red floral top") → direct retrieval.
  - **Recommendation queries** (e.g., "suggest a top for summer") → full pipeline.
- Sub-types for personalized prompts:
  - `GENERAL_TOP_RECOMMENDATION`
  - `TOP_FOR_LOWER_RECOMMENDATION`
  - `SIMILAR_IMAGE_TOP_RECOMMENDATION`



### 2. Similar Top Finder from Image

- Compares user-provided top images using:
  - **Visual attribute matching** (via Gemini).
  - **CLIP/BLIP** image embeddings.
- Gemini-based attribute comparison consistently outperforms in interpretability and accuracy.



### 3. Outfit Builder

- Builds entire outfits around a chosen top:
  - Suggests bottoms, outerwear, accessories, and footwear.
- Factors in:
  - **Color harmony**
  - **Style cohesion**
  - **Occasion relevance**
  - **Body shape compatibility**
  - **Material synergy**
- Uses the same rule-based fashion knowledge base for consistent and stylish outfit generation.



Designed for modular integration into virtual try-on systems, e-commerce platforms, and fashion assistant tools.



## ADDITIONAL FEATURES

### 1) For text-prompt generation
Given a predefined mask placeholder on a neutral background, users provide a textual description of cloth they want to generate(e.g., "a navy-blue polka dot A-line skirt with scalloped hem") to guide the infill model. During inference, the masked region is passed to the diffusion inpainting pipeline alongside the text prompt; the network fills the mask area with a new garment design adhering to the described style.

### 2) For hand-drawn image generation
Users can provide a hand-drawn image of a garment. This image is then analyzed to create a textual description of its features, which is then fed into the same Stable Diffusion Infill Model described above. Using infilling allows the model to easily retain the basic shape of the hand-drawn dress, only needing to add the specific cloth details and textures.


