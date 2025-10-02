import os
import base64
import io
import time # Import the time module for delays
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from pdf2image import convert_from_path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
# Removed botocore.exceptions as it's specific to AWS and not needed for Google Gemini

# Removed ocr_pdf_to_text as it's no longer needed with Bedrock image processing

def ocr_pdf_to_image(pdf_path: str, image_name: str) -> list[str]:
    """
    Converts a PDF to images and saves them as PNG files.
    Returns a list of paths to the saved image files.
    Requires Poppler to be installed on the system.
    """
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_filename = f"{image_name}_page_{i:03d}.png"
        image_path = os.path.join('./', image_filename)
        image.save(image_path, 'PNG')
        print(f"Saved image for page {i} to: {image_path}")
        image_paths.append(image_path)
    return image_paths

def extract_text_from_images_with_gemini(image_paths: list[str], llm: ChatGoogleGenerativeAI) -> str:
    """
    Extracts text from a list of image files using ChatGoogleGenerativeAI's multimodal capabilities.
    """
    full_text = ""
    for i, image_path in enumerate(image_paths):
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            (
                "human",
                [
                    {"type": "text", "text": f"Extract all text from this image. This is page {i+1}."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            )
        ]
        print(f"Extracting text from {image_path} using ChatGoogleGenerativeAI...")
        try:
            ai_msg = llm.invoke(messages)
            page_text = ai_msg.content
            print(f"--- Extracted text from Page {i+1} ---\n{page_text[:200]}...\n") # Print first 200 chars
            full_text += f"--- Page {i+1} ---\n{page_text}\n\n"
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            full_text += f"--- Page {i+1} (Error) ---\n[Text extraction failed]\n\n"
        time.sleep(5) # Add a delay to prevent throttling
    return full_text

# 1. Convert PDF to images
pdf_file_path = "9787480194.pdf"
image_prefix = os.path.splitext(os.path.basename(pdf_file_path))[0]
image_paths = ocr_pdf_to_image(pdf_file_path, image_prefix)

# 2. Set up Langchain LLM with Google Gemini
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual Google Gemini API key.
# For security, consider setting this as an environment variable instead of hardcoding.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Extract text from images using ChatGoogleGenerativeAI
ocr_text = extract_text_from_images_with_gemini(image_paths, llm)

# 4. Implement text splitting
# We'll split the document into smaller chunks to be processed by the map-reduce chain.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
docs = [Document(page_content=x) for x in text_splitter.split_text(ocr_text)]

# 5. Build and execute the Map-Reduce summarization chain
# The 'map_reduce' chain type first summarizes each chunk (map step)
# and then combines these summaries into a final summary (reduce step).
map_prompt_template = """請從以下文本中提取所有商品編號 (Product Number) 和其出產的國家 (Country of Origin)。如果有多個，請列出所有。請以清晰的列表格式呈現。
注意:你必須使用1-1對應
範例格式：
[商品編號]: [出產國家]

文本:
"{text}"
"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """請將以下所有提取到的商品編號和出產國家合併成一個最終的列表。如果有多個，請列出所有。請以清晰的列表格式呈現。
注意:
1. 你必須使用1-1對應
2. 開頭為60 or 6Y的才是我們要的產品編號


範例格式：
[商品編號]: [出產國家]

提取到的資訊:
"{text}"
"""
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, verbose=True)

@retry(
    wait=wait_exponential(multiplier=1, min=10, max=60),
    stop=stop_after_attempt(7), # Reverted to 7 attempts
    reraise=True,
)
def run_chain_with_retries(chain, docs):
    print("Attempting extraction with Map-Reduce chain...")
    return chain.run(docs)

output_result = run_chain_with_retries(chain, docs)

# 6. Display the results
print("\n--- Extracted Information from PDF ---")
print(output_result)

# 7. Clean up temporary image files
for image_path in image_paths:
    try:
        os.remove(image_path)
        print(f"Removed temporary image: {image_path}")
    except OSError as e:
        print(f"Error removing temporary image {image_path}: {e}")
