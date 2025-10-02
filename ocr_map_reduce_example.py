import os
from langchain_aws import ChatBedrock
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from pdf2image import convert_from_path
import pytesseract
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import botocore.exceptions

def ocr_pdf_to_text(pdf_path: str) -> str:
    """
    Converts a PDF to images, performs OCR on each image, and returns the concatenated text.
    Requires Poppler and Tesseract to be installed on the system.
    """
    images = convert_from_path(pdf_path)
    full_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='eng') # You can specify other languages if needed
        print(text)
        full_text += f"--- Page {i+1} ---\n{text}\n\n"
    return full_text


def ocr_pdf_to_image(pdf_path: str, image_name: str) -> str:
    """
    Converts a PDF to images, performs OCR on each image, and returns the concatenated text.
    Requires Poppler and Tesseract to be installed on the system.
    """
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        image_filename = f"{image_name}_page_{i:03d}.png"
        image_path = os.path.join('./', image_filename)
        image.save(image_path, 'PNG')
        print(f"Saved image for page {i} to: {image_path}")
    return images

# 1. Perform OCR on the PDF file
pdf_file_path = "9787480194.pdf"
# ocr_text = ocr_pdf_to_text(pdf_file_path)


# 2. Set up Langchain LLM with AWS Bedrock
# Ensure your AWS credentials are configured (e.g., via environment variables or ~/.aws/credentials)
# The region name should match where your Bedrock model is available.
llm = ChatBedrock(
    # model_id="anthropic.claude-3-sonnet-20240229-v1:0", # Updated model ID for Claude 3 Sonnet
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name = "ap-southeast-2"
    # region_name="us-east-1"
)
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)

'''
# 3. Implement text splitting
# We'll split the document into smaller chunks to be processed by the map-reduce chain.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
docs = [Document(page_content=x) for x in text_splitter.split_text(ocr_text)]

# 4. Build and execute the Map-Reduce summarization chain
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
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(7),
    retry=retry_if_exception_type(botocore.exceptions.ClientError),
    reraise=True,
)
def run_chain_with_retries(chain, docs):
    print("Attempting extraction with Map-Reduce chain...")
    return chain.run(docs)

output_result = run_chain_with_retries(chain, docs)

# 5. Display the results
print("\n--- Extracted Information from PDF ---")
print(output_result)
'''