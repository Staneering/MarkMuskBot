import json
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
import os

# FastAPI initialization
app = FastAPI()

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load GPT-4 for code generation
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model_name="gpt-4")

def scrape_page(url):
    """
    Scrapes the content of a single page and returns the structured text.

    Args:
        url (str): The URL of the page to scrape.

    Returns:
        str: The structured text content of the page, or None on error.
    """
    try:
        print(f"Scraping URL: {url}")
        response = requests.get("https://docs.creditchek.africa" + url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the main content section of the page (adjust as needed based on website structure)
        main_content = soup.find('div', {'class': 'main-wrapper'})  # Update selector as needed

        if not main_content:
            print(f"Warning: Could not find main content on {url}")
            return None

        # Extract all text, preserving some structure
        text_parts = []
        for element in main_content.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'pre']):  # Extract headings, paragraphs, lists, code blocks
            if element.name in ['h1', 'h2', 'h3']:
                text_parts.append(f"\n{element.text.strip()}\n")
            elif element.name == 'p':
                text_parts.append(f"{element.text.strip()}\n")
            elif element.name == 'pre':  # Preserve formatting for code blocks
                text_parts.append(f"\n```\n{element.text.strip()}\n```\n")
            elif element.name in ['ul', 'ol']:  # Handle lists
                items = [f"  * {item.text.strip()}" for item in element.find_all('li')]
                text_parts.append("\n" + "\n".join(items) + "\n")
            else:
                text_parts.append(element.text.strip())

        structured_text = "".join(text_parts).strip()
        return structured_text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None


def scrape_navbar_and_pages(navbar_data):
    """
    Scrapes the content of each page in the navbar data and returns a list of dictionaries.

    Args:
        navbar_data (list): A list of dictionaries representing the navbar structure.

    Returns:
        list: A list of dictionaries, where each dictionary contains the page URL,
              title, and structured text content.
    """
    results = []
    
    def process_item(item):
        """Helper function to recursively process navbar items."""
        page_data = {}
        url = item.get('link')
        label = item.get('label')

        if url:
            full_url = url  # Assumes relative URLs. Modify if needed for full URLs.
            structured_text = scrape_page(full_url)
            if structured_text:
                page_data = {
                    'url': full_url,
                    'title': label,
                    'content': structured_text,
                }
                results.append(page_data)

        if 'children' in item:
            for child in item['children']:
                process_item(child)

    for item in navbar_data:
        process_item(item)
    
    return results


def generate_embeddings_and_save(results):
    """
    Generates embeddings for the scraped content and stores them in a FAISS index.

    Args:
        results (list): List of dictionaries containing the page content.
    """
    # Prepare the documents (scraped content from the previous step)
    documents = [page['content'] for page in results]
    titles = [page['title'] for page in results]

    # Generate embeddings
    embeddings = model.encode(documents)

    # Convert embeddings to a FAISS index for efficient semantic search
    dimension = embeddings.shape[1]  # The embedding dimension (length of the vector)
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    # Save the index to a file
    faiss.write_index(index, 'creditchek_index.faiss')

    # Save the titles and documents for reference
    with open("titles.json", "w") as f:
        json.dump(titles, f)

    with open("documents.json", "w") as f:
        json.dump(documents, f)


class QueryRequest(BaseModel):
    query: str
    language: str = "python"  # Default to Python


async def generate_response_from_query(query: str):
    """Helper function to generate responses using FAISS search and OpenAI's GPT-4"""
    try:
        # Generate embedding for the query
        query_embedding = model.encode([query])

        # Load the FAISS index and titles/documents
        index = faiss.read_index('creditchek_index.faiss')
        with open('titles.json', 'r') as f:
            titles = json.load(f)
        with open('documents.json', 'r') as f:
            documents = json.load(f)

        # Search the FAISS index for the most relevant documentation
        D, I = index.search(np.array(query_embedding, dtype=np.float32), k=3)  # k=3 for top 3 results

        # Get the titles of the top 3 documents
        top_docs = [titles[i] for i in I[0]]

        # Generate response using OpenAI's GPT-4 to answer the query based on the top documents
        docs_text = "\n".join([documents[i] for i in I[0]])
        response = llm(f"Answer the following query using these documents: {docs_text}\n\nQuery: {query}")
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(request: QueryRequest):
    """Handles developer queries and provides API documentation or code samples."""
    try:
        response = await generate_response_from_query(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-code")
async def generate_code(request: QueryRequest):
    """Generates API integration code in the specified programming language."""
    try:
        # Generate a code sample using OpenAI's GPT-4 based on the query
        code_query = f"Generate a {request.language} API integration code snippet for: {request.query}"
        response = await generate_response_from_query(code_query)
        return {"code": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    # Your navbar data (from the immersive artifact)
    navbar_data = [
        {
            "label": "Introduction",
            "link": "/intro",
            "level": 1,
            "active": True
        },
        {
            "label": "Authentication",
            "link": "/auth",
            "level": 1
        },
        {
            "label": "Webhooks",
            "link": "/webhook",
            "level": 1
        },
        {
            "label": "Credit Assessment SDK",
            "link": "/category/credit-assessment-sdk",
            "level": 1,
            "children": [
                {
                    "label": "Overview",
                    "link": "/widget/overview",
                    "level": 2
                },
                {
                    "label": "Get Started",
                    "link": "/category/get-started",
                    "level": 2,
                    "children": [
                        {
                            "label": "Client-side Rendering",
                            "link": "/widget/getStarted/clientSide",
                            "level": 3
                        },
                        {
                            "label": "Server-side Implementation",
                            "link": "/widget/getStarted/serverSide",
                            "level": 3
                        }
                    ]
                },
                {
                    "label": "Sample Application",
                    "link": "/widget/sampleApplication",
                    "level": 2
                }
            ]
        },
        {
            "label": "Nigeria 🇳🇬",
            "link": "/category/nigeria-",
            "level": 1,
            "children": [
                {
                    "label": "Identity Service",
                    "link": "/category/identity-service",
                    "level": 2,
                    "children": [
                        {
                            "label": "Get base URL",
                            "link": "/nigeria/identity/baseUrl",
                            "level": 3
                        },
                        {
                            "label": "Submit borrower data (consent & onboarding link)",
                            "link": "/nigeria/identity/submitBorrower",
                            "level": 3
                        },
                        {
                            "label": "Get borrower",
                            "link": "/nigeria/identity/getBorrower",
                            "level": 3
                        },
                        {
                            "label": "Delete borrower",
                            "link": "/nigeria/identity/deleteBorrower",
                            "level": 3
                        },
                        {
                            "label": "BVN Identity Verification",
                            "link": "/nigeria/identity/bvnVerification",
                            "level": 3
                        },
                        {
                            "label": "KYC Consent(BVN-iGree)",
                            "link": "/nigeria/identity/bvnIgree",
                            "level": 3
                        },
                        {
                            "label": "NIN Identity Verification",
                            "link": "/nigeria/identity/ninVerification",
                            "level": 3
                        },
                        {
                            "label": "CAC RC Number Verification",
                            "link": "/nigeria/identity/cacVerification",
                            "level": 3
                        },
                        {
                            "label": "Bank Account Number Verification (Basic)",
                            "link": "/nigeria/identity/accountVerification",
                            "level": 3
                        },
                        {
                            "label": "Bank Account Number Verification (Advanced)",
                            "link": "/nigeria/identity/advancedAccountVerificatioon",
                            "level": 3
                        },
                        {
                            "label": "Driver's License Verification",
                            "link": "/nigeria/identity/driverVerification",
                            "level": 3
                        },
                        {
                            "label": "International Passport Verification",
                            "link": "/nigeria/identity/passportVerification",
                            "level": 3
                        }
                    ]
                },
                {
                    "label": "Income Service",
                    "link": "/category/income-service",
                    "level": 2,
                    "children": [
                        {
                            "label": "Connect Bank Account",
                            "link": "/nigeria/income/connectAccounts",
                            "level": 3
                        },
                        {
                            "label": "Get base URL",
                            "link": "/nigeria/income/baseUrl",
                            "level": 3
                        },
                        {
                            "label": "Open banking",
                            "link": "/category/open-banking",
                            "level": 3,
                            "children": [
                                {
                                    "label": "Step 1: Get list of banks supported",
                                    "link": "/nigeria/income/openBanking/getBanks",
                                    "level": 4
                                },
                                {
                                    "label": "Step 1: Initialize consent",
                                    "link": "/nigeria/income/openBanking/initializeConsent",
                                    "level": 4
                                },
                                {
                                    "label": "Step 2: Fetch transactions",
                                    "link": "/nigeria/income/openBanking/getTransactions",
                                    "level": 4
                                }
                            ]
                        },
                        {
                            "label": "DBR Keywords",
                            "link": "/nigeria/income/dbr",
                            "level": 3
                        },
                        {
                            "label": "Upload Bank Statement PDF",
                            "link": "/nigeria/income/uploadPdf",
                            "level": 3
                        },
                        {
                            "label": "Get Income Insights",
                            "link": "/nigeria/income/incomeInsights",
                            "level": 3
                        },
                        {
                            "label": "Get Existing Income Insights",
                            "link": "/nigeria/income/getExistingInsightData",
                            "level": 3
                        },
                        {
                            "label": "Get borrower's linked accounts",
                            "link": "/nigeria/income/getBorrowerAccounts",
                            "level": 3
                        },
                        {
                            "label": "Get All Linked Accounts",
                            "link": "/nigeria/income/getAllLinkedAccount",
                            "level": 3
                        },
                        {
                            "label": "Delete Linked Account",
                            "link": "/nigeria/income/deleteLinkedAccount",
                            "level": 3
                        },
                        {
                            "label": "Webhooks",
                            "link": "/nigeria/income/webhook",
                            "level": 3
                        }
                    ]
                },
                {
                    "label": "Credit Insights",
                    "link": "/category/credit-insights",
                    "level": 2,
                    "children": [
                        {
                            "label": "Get base URL",
                            "link": "/nigeria/credit/baseUrl",
                            "level": 3
                        },
                        {
                            "label": "For Individuals",
                            "link": "/category/for-individuals",
                            "level": 3,
                            "children": [
                                {
                                    "label": "CRC",
                                    "link": "/category/crc",
                                    "level": 4,
                                    "children": [
                                        {
                                            "label": "CRC",
                                            "link": "/nigeria/credit/individuals/crc/",
                                            "level": 5
                                        },
                                        {
                                            "label": "CRC FICO Scores",
                                            "link": "/nigeria/credit/individuals/crc/crcFico",
                                            "level": 5
                                        },
                                        {
                                            "label": "CRC Full Report",
                                            "link": "/nigeria/credit/individuals/crc/crcPremium",
                                            "level": 5
                                        }
                                    ]
                                },
                                {
                                    "label": "Credit Registry",
                                    "link": "/category/credit-registry",
                                    "level": 4,
                                    "children": [
                                        {
                                            "label": "Credit Registry",
                                            "link": "/nigeria/credit/individuals/credit registry/creditRegistry",
                                            "level": 5
                                        },
                                        {
                                            "label": "Credit Registry Full Report",
                                            "link": "/nigeria/credit/individuals/credit registry/creditRegistryPremium",
                                            "level": 5
                                        }
                                    ]
                                },
                                {
                                    "label": "First Central",
                                    "link": "/category/first-central",
                                    "level": 4,
                                    "children": [
                                        {
                                            "label": "First Central",
                                            "link": "/nigeria/credit/individuals/first central/firstCentral",
                                            "level": 5
                                        },
                                        {
                                            "label": "First Central I-Score",
                                            "link": "/nigeria/credit/individuals/first central/firstCentralIscore",
                                            "level": 5
                                        },
                                        {
                                            "label": "First Central Full Report",
                                            "link": "/nigeria/credit/individuals/first central/firstCentralPremium",
                                            "level": 5
                                        }
                                    ]
                                },
                                {
                                    "label": "Advanced",
                                    "link": "/nigeria/credit/individuals/advanced",
                                    "level": 4
                                },
                                {
                                    "label": "Premium",
                                    "link": "/nigeria/credit/individuals/premium",
                                    "level": 4
                                }
                            ]
                        },
                        {
                            "label": "For Businesses",
                            "link": "/category/for-businesses",
                            "level": 3,
                            "children": [
                                {
                                    "label": "SME CRC",
                                    "link": "/nigeria/credit/business/smeCrc",
                                    "level": 4
                                },
                                {
                                    "label": "SME First Central",
                                    "link": "/nigeria/credit/business/smeFirstCentral",
                                    "level": 4
                                },
                                {
                                    "label": "SME Premium",
                                    "link": "/nigeria/credit/business/smePremium",
                                    "level": 4
                                }
                            ]
                        },
                        {
                            "label": "NANO Specials",
                            "link": "/category/nano-specials",
                            "level": 3,
                            "children": [
                                {
                                    "label": "CRC Nano",
                                    "link": "/nigeria/credit/nano/crcNano",
                                    "level": 4
                                },
                                {
                                    "label": "First Central Nano",
                                    "link": "/nigeria/credit/nano/firstCentralNano",
                                    "level": 4
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    # Scrape the navbar and pages
    results = scrape_navbar_and_pages(navbar_data)

    # Generate embeddings and save the FAISS index
    generate_embeddings_and_save(results)

    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
