import asyncio
import os
import re
from typing import Annotated

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import google.generativeai as genai
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS
from pydantic import BaseModel, Field

import markdownify
import httpx

# --- Load environment variables ---
load_dotenv()

# Puch & Auth Tokens
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

# Google Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

# --- Assertions for required environment variables ---
assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"


# --- Auth Provider (Required by Puch) ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    """A simple bearer token authentication provider for the server."""
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token, client_id="puch-client", scopes=["*"], expires_at=None
            )
        return None


# --- Rich Tool Description model (for better AI understanding) ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


# --- Web Content Fetching and Parsing Utility Class ---
class WebFetcher:
    """A utility class to fetch and parse web content."""
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Puch/1.0"

    @classmethod
    async def fetch_html(cls, url: str) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url, follow_redirects=True, headers={"User-Agent": cls.USER_AGENT}, timeout=30
                )
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

    @staticmethod
    def parse_html_to_markdown(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        if not main_content:
            return "<error>Could not find main content in the page.</error>"
        markdown_text = markdownify.markdownify(str(main_content), heading_style=markdownify.ATX)
        return markdown_text.strip()

# --- Advanced Multi-Agent Analysis Pipeline (Now using Gemini) ---
class AnalysisPipeline:
    """Handles the advanced multi-agent logic for analyzing large texts."""
    def __init__(self, api_key, model_name):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def create_chunks(self, text: str, chunk_size: int = 15000, overlap: int = 500) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    async def run_triage_agent(self, chunk: str, original_query: str) -> bool:
        if "summarize" in original_query.lower():
            return True
        prompt = (
            "You are a Triage Assistant. Your task is to determine if the given text chunk contains information that is potentially relevant to the user's question. "
            "Focus on keywords and concepts. Answer only with 'YES' if it is relevant, or 'NO' if it is not.Think hard about the user's question and the content of the chunk."
            "Assign relevance labels carefully, even if little relevance is found."
            "\n\n"
            f"User's question: '{original_query}'\n\n"
            f"Text chunk to analyze:\n---\n{chunk}\n---"
        )
        try:
            response = await self.model.generate_content_async(prompt)
            return "YES" in response.text.strip().upper()
        except Exception as e:
            return False

    async def run_summarization_agent(self, chunk: str, original_query: str) -> str:
        prompt = (
            "You are a Data Extraction Specialist. Your task is to carefully read the provided text and extract all facts, figures, and key points that are directly relevant to the user's question. "
            "Present the extracted information as a concise, dense summary. If no relevant information is found, respond with an empty string.\n\n"
            f"User's question: '{original_query}'\n\n"
            f"Text to analyze:\n---\n{chunk}\n---"
        )
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return ""

    async def run_synthesizer_agent(self, relevant_summaries: list[str], original_query: str) -> str:
        if not relevant_summaries:
            return "I scanned the document but could not find any information relevant to your question."
        combined_context = "\n\n---\n\n".join(relevant_summaries)
        prompt = (
            "You are a Final Answer Synthesizer. You have been provided with a collection of relevant text summaries extracted from a long document. "
            "Your task is to synthesize this information into a single, comprehensive, and well-written answer to the user's original question. "
            "Base your answer ONLY on the provided summaries. Do not add outside information or repeat yourself."
            "Answer only the question asked, and do not include any additional information or context."
            "Don't include any suggestions or prompts for further action."
            "\n\n"
            f"User's original question: '{original_query}'\n\n"
            f"Combined relevant information:\n---\n{combined_context}\n---"
        )
        response = await self.model.generate_content_async(prompt)
        return response.text


# --- MCP Server Setup ---
# --- CHANGE: Moved stateless_http=True to the constructor to fix the Vercel crash ---
mcp = FastMCP(
    "Web Analyzer MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
    stateless_http=True
)

# --- This is the line that exposes the app for Vercel ---
app = mcp.streamable_http_app()

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    """Validates the server's identity for Puch."""
    return MY_NUMBER


# --- Main Web Scraping and Analysis Tool ---
WebAnalyzerDescription = RichToolDescription(
    description="Analyzes a webpage's content to answer user questions. It can handle very long pages by intelligently processing them in parts.",
    use_when="**MUST USE** this tool if the user's query contains a URL (e.g., 'http://', 'https://', 'www.'). This tool is specifically designed to handle web links.",
    side_effects="Makes multiple external network requests to the provided URL and to an AI service. This may take several seconds for long pages.",
)

@mcp.tool(
    description=WebAnalyzerDescription.model_dump_json()
)
async def web_analyzer(
    user_query: Annotated[str, Field(description="The user's full query, which should include a URL and a question.")]
) -> str:
    """
    Extracts a URL from the user's query, scrapes it, and uses a multi-agent AI pipeline to answer the query based on the page content.
    """
    url_match = re.search(r'https?://[^\s]+', user_query)
    
    if not url_match:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="I couldn't find a URL in your request. Please provide a full web address."))

    url = url_match.group(0)
    parts = re.split(re.escape(url), user_query)
    query_text = ' '.join(parts).strip()
    if not query_text:
        query_text = "Summarize the content of the page."

    try:
        html_content = await WebFetcher.fetch_html(url)
        markdown_content = WebFetcher.parse_html_to_markdown(html_content)
        
        pipeline = AnalysisPipeline(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
        
        chunks = pipeline.create_chunks(markdown_content)

        BATCH_SIZE = 5
        DELAY_BETWEEN_BATCHES = 1

        # Triage Agent: Run in batches
        triage_results = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            batch_tasks = [pipeline.run_triage_agent(chunk, query_text) for chunk in batch_chunks]
            results = await asyncio.gather(*batch_tasks)
            triage_results.extend(results)
            if i + BATCH_SIZE < len(chunks):
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)
        
        relevant_chunks = [chunks[i] for i, is_relevant in enumerate(triage_results) if is_relevant]

        if not relevant_chunks:
            return "I scanned the document but could not find any information relevant to your question."

        # Summarization Agent: Run in batches
        chunk_summaries = []
        for i in range(0, len(relevant_chunks), BATCH_SIZE):
            batch_chunks = relevant_chunks[i:i + BATCH_SIZE]
            batch_tasks = [pipeline.run_summarization_agent(chunk, query_text) for chunk in batch_chunks]
            results = await asyncio.gather(*batch_tasks)
            chunk_summaries.extend(results)
            if i + BATCH_SIZE < len(relevant_chunks):
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)
        
        valid_summaries = [summary for summary in chunk_summaries if summary.strip()]
        
        # Synthesizer Agent: Get the final answer
        final_answer = await pipeline.run_synthesizer_agent(valid_summaries, query_text)
        
        return final_answer

    except McpError as e:
        raise e
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"An unexpected error occurred: {str(e)}"))

# --- The main() function and __main__ block have been removed for Vercel deployment ---
