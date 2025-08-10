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

# Starlette to mount MCP app at /mcp for Vercel
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route, Mount

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-latest")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None

# --- Models / Utilities ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

class WebFetcher:
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Puch/1.0"
    @classmethod
    async def fetch_html(cls, url: str) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, follow_redirects=True, headers={"User-Agent": cls.USER_AGENT}, timeout=30)
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

# --- Analysis pipeline (Gemini) ---
class AnalysisPipeline:
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
            "You are a Triage Assistant. Answer only 'YES' or 'NO'.\n\n"
            f"User's question: '{original_query}'\n\nText chunk:\n---\n{chunk}\n---"
        )
        try:
            response = await self.model.generate_content_async(prompt)
            return "YES" in (response.text or "").strip().upper()
        except Exception:
            return False

    async def run_summarization_agent(self, chunk: str, original_query: str) -> str:
        prompt = (
            "You are a Data Extraction Specialist. Extract facts relevant to the question.\n\n"
            f"User's question: '{original_query}'\n\nText:\n---\n{chunk}\n---"
        )
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text or ""
        except Exception:
            return ""

    async def run_synthesizer_agent(self, relevant_summaries: list[str], original_query: str) -> str:
        if not relevant_summaries:
            return "I scanned the document but could not find any information relevant to your question."
        combined_context = "\n\n---\n\n".join(relevant_summaries)
        prompt = (
            "You are a Final Answer Synthesizer. Base your answer ONLY on provided summaries.\n\n"
            f"User's original question: '{original_query}'\n\nCombined:\n---\n{combined_context}\n---"
        )
        response = await self.model.generate_content_async(prompt)
        return response.text or ""

# --- MCP server and tools ---
mcp = FastMCP("Web Analyzer MCP Server", auth=SimpleBearerAuthProvider(TOKEN), stateless_http = True)

@mcp.tool
async def validate() -> str:
    return MY_NUMBER

WebAnalyzerDescription = RichToolDescription(
    description="Analyzes webpage content for user queries; handles long pages by chunking.",
    use_when="Use when user's query contains a URL.",
    side_effects="Makes external requests to the target and to Gemini."
)

@mcp.tool(description=WebAnalyzerDescription.model_dump_json())
async def web_analyzer(user_query: Annotated[str, Field(description="Full query including a URL")]) -> str:
    url_match = re.search(r'https?://[^\s]+', user_query)
    if not url_match:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="I couldn't find a URL in your request. Please provide a full web address."))
    url = url_match.group(0)
    parts = re.split(re.escape(url), user_query)
    query_text = ' '.join(parts).strip() or "Summarize the content of the page."
    try:
        html_content = await WebFetcher.fetch_html(url)
        markdown_content = WebFetcher.parse_html_to_markdown(html_content)
        pipeline = AnalysisPipeline(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
        chunks = pipeline.create_chunks(markdown_content)
        BATCH_SIZE = 5
        DELAY_BETWEEN_BATCHES = 1

        triage_results = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            results = await asyncio.gather(*[pipeline.run_triage_agent(c, query_text) for c in batch_chunks])
            triage_results.extend(results)
            if i + BATCH_SIZE < len(chunks):
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

        relevant_chunks = [chunks[i] for i, is_relevant in enumerate(triage_results) if is_relevant]
        if not relevant_chunks:
            return "I scanned the document but could not find any information relevant to your question."

        chunk_summaries = []
        for i in range(0, len(relevant_chunks), BATCH_SIZE):
            batch_chunks = relevant_chunks[i:i + BATCH_SIZE]
            results = await asyncio.gather(*[pipeline.run_summarization_agent(c, query_text) for c in batch_chunks])
            chunk_summaries.extend(results)
            if i + BATCH_SIZE < len(relevant_chunks):
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

        valid_summaries = [s for s in chunk_summaries if s.strip()]
        final_answer = await pipeline.run_synthesizer_agent(valid_summaries, query_text)
        return final_answer

    except McpError as e:
        raise e
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"An unexpected error occurred: {str(e)}"))

# ---------------- Vercel-friendly ASGI export ----------------
# Use non-streamable "http" transport (no background task group required)
# --- FINAL FIX: Set the path to "/" since Starlette handles the "/mcp/" prefix ---
# mcp_asgi = mcp.http_app(transport="http", path="/")
# if mcp_asgi is None:
#     raise RuntimeError("mcp.http_app(...) returned None. Check fastmcp version and usage.")

# async def root(request):
#     return PlainTextResponse("MCP server running. Use POST /mcp/ with proper auth headers.")

# routes = [
#     Route("/", root),
#     Mount("/mcp/", mcp_asgi),
# ]

app = mcp.http_app(transport="streamable-http")
