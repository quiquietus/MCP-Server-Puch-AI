# main.py
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

# Load env from .env (locally). On Vercel, set real env vars in dashboard.
load_dotenv()

# Required environment variables
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file or environment"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file or environment"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file or environment"

# --- Auth Provider (Puch requirement) ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None

# --- Tool description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Web fetching / parsing utility ---
class WebFetcher:
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Puch/1.0"
    )

    @classmethod
    async def fetch_html(cls, url: str) -> str:
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, follow_redirects=True, headers={"User-Agent": cls.USER_AGENT}, timeout=30)
                resp.raise_for_status()
                return resp.text
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

    @staticmethod
    def parse_html_to_markdown(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg", "iframe"]):
            tag.decompose()
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        if not main_content:
            return "<error>Could not find main content in the page.</error>"
        md = markdownify.markdownify(str(main_content), heading_style=markdownify.ATX)
        return md.strip()

# --- Multi-agent analysis pipeline using Google Gemini ---
class AnalysisPipeline:
    def __init__(self, api_key: str, model_name: str):
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
        # quick path for "summarize" queries
        if "summarize" in original_query.lower():
            return True

        prompt = (
            "You are a Triage Assistant. Answer ONLY with YES or NO. "
            "Return YES if the chunk is relevant to the user's question, otherwise NO.\n\n"
            f"User's question: {original_query}\n\n"
            f"Chunk:\n---\n{chunk}\n---"
        )
        try:
            resp = await self.model.generate_content_async(prompt)
            return "YES" in (resp.text or "").strip().upper()
        except Exception:
            return False

    async def run_summarization_agent(self, chunk: str, original_query: str) -> str:
        prompt = (
            "You are a Data Extraction Specialist. Extract facts and key points directly relevant "
            "to the user's question. If none, return an empty string.\n\n"
            f"User's question: {original_query}\n\n"
            f"Text:\n---\n{chunk}\n---"
        )
        try:
            resp = await self.model.generate_content_async(prompt)
            return resp.text or ""
        except Exception:
            return ""

    async def run_synthesizer_agent(self, relevant_summaries: list[str], original_query: str) -> str:
        if not relevant_summaries:
            return "I scanned the document but could not find any information relevant to your question."

        combined = "\n\n---\n\n".join(relevant_summaries)
        prompt = (
            "You are a Final Answer Synthesizer. Use ONLY the provided summaries to answer the user's question. "
            "Do not invent facts. Answer concisely.\n\n"
            f"User's question: {original_query}\n\n"
            f"Summaries:\n---\n{combined}\n---"
        )
        resp = await self.model.generate_content_async(prompt)
        return resp.text or "No useful content extracted."

# --- MCP server setup (stateless) ---
mcp = FastMCP("Web Analyzer MCP Server", auth=SimpleBearerAuthProvider(TOKEN), stateless_http=True)

# Create the ASGI app using a stateless/simple transport (no streaming)
# This returns a FastAPI app that you can deploy to Vercel or other serverless platforms.
app = mcp.http_app(transport="simple-http")

# --- Required validate tool ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- web_analyzer tool (keeps same signature expected by Puch) ---
WebAnalyzerDescription = RichToolDescription(
    description="Analyze a webpage referenced in the user's query and answer questions about it.",
    use_when="When the user's message contains a URL and asks a question about that page.",
    side_effects="Makes network requests and calls Google Gemini."
)

@mcp.tool(description=WebAnalyzerDescription.model_dump_json())
async def web_analyzer(user_query: Annotated[str, Field(description="User query including a URL")]) -> str:
    url_match = re.search(r"https?://[^\s]+", user_query or "")
    if not url_match:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="No URL found in the provided query."))

    url = url_match.group(0)
    # infer the question text by removing the URL from the input
    parts = re.split(re.escape(url), user_query)
    query_text = " ".join([p.strip() for p in parts if p.strip()]).strip()
    if not query_text:
        query_text = "Summarize the content of the page."

    try:
        html = await WebFetcher.fetch_html(url)
        md = WebFetcher.parse_html_to_markdown(html)

        pipeline = AnalysisPipeline(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
        chunks = pipeline.create_chunks(md)

        BATCH_SIZE = 5
        DELAY = 1

        # Triage
        triage_results = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            tasks = [pipeline.run_triage_agent(c, query_text) for c in batch]
            res = await asyncio.gather(*tasks)
            triage_results.extend(res)
            if i + BATCH_SIZE < len(chunks):
                await asyncio.sleep(DELAY)

        relevant_chunks = [chunks[i] for i, ok in enumerate(triage_results) if ok]
        if not relevant_chunks:
            return "I scanned the document but could not find any information relevant to your question."

        # Summarize relevant chunks
        chunk_summaries = []
        for i in range(0, len(relevant_chunks), BATCH_SIZE):
            batch = relevant_chunks[i : i + BATCH_SIZE]
            tasks = [pipeline.run_summarization_agent(c, query_text) for c in batch]
            res = await asyncio.gather(*tasks)
            chunk_summaries.extend(res)
            if i + BATCH_SIZE < len(relevant_chunks):
                await asyncio.sleep(DELAY)

        valid_summaries = [s for s in chunk_summaries if s and s.strip()]
        final = await pipeline.run_synthesizer_agent(valid_summaries, query_text)
        return final

    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error: {e!s}"))
