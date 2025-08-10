# demoServer.py
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
import asyncio

# --- Load environment variables (local dev) ---
load_dotenv()

# --- Required env vars (set these in Vercel Dashboard for production) ---
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file / Vercel env"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file / Vercel env"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file / Vercel env"

# --- Simple Bearer auth provider (keeps Puch starter format) ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        # keep compatibility: provide public_key, jwks_uri None
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None

# --- A tiny descriptive model for tools (optional, but useful) ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Web fetch & parse utilities ---
class WebFetcher:
    USER_AGENT = "Puch/1.0 (web-analyzer)"

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
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return "<error>Could not find main content</error>"
        md = markdownify.markdownify(str(main), heading_style=markdownify.ATX)
        return md.strip()

# --- Multi-agent analysis pipeline using Gemini ---
class AnalysisPipeline:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        # create a model handle; wrapper API may vary by gemini client version
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
            "You are a Triage Assistant — answer with only YES or NO whether the chunk is relevant.\n\n"
            f"Question: {original_query}\n\nChunk:\n---\n{chunk[:4000]}\n---\nAnswer YES or NO:"
        )
        try:
            # async generate - API naming can change between SDK versions
            resp = await self.model.generate_content_async(prompt)
            return "YES" in (resp.text or "").upper()
        except Exception:
            return False

    async def run_summarization_agent(self, chunk: str, original_query: str) -> str:
        prompt = (
            "Extract concise key facts from the text that directly answer the user's question. "
            "If nothing relevant, return an empty string.\n\n"
            f"Question: {original_query}\n\nText:\n---\n{chunk[:14000]}\n---"
        )
        try:
            resp = await self.model.generate_content_async(prompt)
            return resp.text or ""
        except Exception:
            return ""

    async def run_synthesizer_agent(self, summaries: list[str], original_query: str) -> str:
        if not summaries:
            return "I scanned the document but could not find any information relevant to your question."
        combined = "\n\n---\n\n".join(summaries)
        prompt = (
            "Synthesize a single concise answer using only the provided summaries. Do NOT hallucinate.\n\n"
            f"Question: {original_query}\n\nContext:\n---\n{combined}\n---"
        )
        resp = await self.model.generate_content_async(prompt)
        return resp.text or "No answer returned."

# --- Build FastMCP and export an ASGI app (Vercel expects `app`) ---
mcp = FastMCP("Web Analyzer MCP Server", auth=SimpleBearerAuthProvider(TOKEN))
# Note: Do NOT pass stateless_http in constructor to avoid lifespan issues.

# required by Puch
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# tool description
WebAnalyzerDescription = RichToolDescription(
    description="Scrape a web page (last URL in the user text) and answer a user's question about it using a multi-agent pipeline (Gemini).",
    use_when="Use when the user's message contains a URL and they ask a question or ask to summarize.",
    side_effects="Performs external HTTP requests and calls Gemini APIs."
)

URL_RE = re.compile(r"https?://[^\s'\"<>]+")

@mcp.tool(description=WebAnalyzerDescription.model_dump_json())
async def web_analyzer(user_query: Annotated[str, Field(description="User text containing a URL and question")]) -> str:
    # extract last URL
    match = URL_RE.findall(user_query or "")
    if not match:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="No URL found in the input. Provide a full URL."))
    url = match[-1].rstrip(").,")
    # infer question text
    parts = re.split(re.escape(url), user_query)
    question = " ".join(parts).strip()
    if not question:
        question = "Summarize the page."

    # fetch
    html = await WebFetcher.fetch_html(url)
    md = WebFetcher.parse_html_to_markdown(html)

    # save local debug file (ephemeral in serverless)
    try:
        with open("/tmp/scraped_content.md", "w", encoding="utf-8") as f:
            f.write(f"--- Source: {url} ---\n\n")
            f.write(md[:200000])  # cap size
    except Exception:
        # ignore file write failures in restricted envs
        pass

    pipeline = AnalysisPipeline(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    chunks = pipeline.create_chunks(md)

    # triage
    triage = []
    BATCH = 4
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        tasks = [pipeline.run_triage_agent(c, question) for c in batch]
        results = await asyncio.gather(*tasks)
        triage.extend(results)
        if i + BATCH < len(chunks):
            await asyncio.sleep(0.5)

    relevant = [chunks[i] for i, ok in enumerate(triage) if ok]
    if not relevant:
        return "I scanned the page but could not find relevant information."

    # summarize relevant
    summaries = []
    for i in range(0, len(relevant), BATCH):
        batch = relevant[i : i + BATCH]
        tasks = [pipeline.run_summarization_agent(c, question) for c in batch]
        results = await asyncio.gather(*tasks)
        summaries.extend([r for r in results if r and r.strip()])
        if i + BATCH < len(relevant):
            await asyncio.sleep(0.5)

    # final synth
    answer = await pipeline.run_synthesizer_agent(summaries, question)
    # prefix source
    header = f"🔗 Source: {url}\n\n"
    return header + answer

# --- Expose ASGI app for Vercel: no other routes, no streaming ---
# Create the ASGI app that serves MCP on /mcp/ (this sets up proper lifespan)
mcp_asgi = mcp.http_app(path="/mcp")

# Export `app` (Vercel's Python runtime expects this)
app = mcp_asgi

# (no __main__, no run())
