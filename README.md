# Web Analyzer MCP Server

## Overview
The **Web Analyzer MCP Server** is a [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) server that processes web pages and answers user queries about them.  
It is designed to:
- Fetch web pages from a given URL.
- Clean and convert the content into Markdown.
- Split the text into manageable chunks.
- Use a **multi-agent pipeline** with Google Gemini to:
  1. **Triage Agent** – Identify relevant chunks for the query.
  2. **Summarization Agent** – Extract key points from relevant chunks.
  3. **Synthesizer Agent** – Combine summaries into a final answer.

The server supports **Bearer token authentication** and integrates seamlessly with tools like **Puch**.

---

## Features
- **Multi-Agent Pipeline**
  - Triage → Summarization → Final Synthesis.
- **Gemini AI Integration**
  - Uses Google’s `gemini-2.0-flash-lite` model (configurable).
- **Markdown Conversion**
  - Strips scripts, styles, headers, footers, and sidebars.

---

## Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Python Version
Python **3.11+** is recommended.

---

## Environment Variables
Create a `.env` file with:

```env
AUTH_TOKEN=YOUR_AUTH_TOKEN
MY_NUMBER=YOUR_PHONE_NUMBER
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-2.0-flash-lite
```

---

## Running the Server
Start the MCP server:

```bash
python demoServer.py
```

It will run on:
```
http://0.0.0.0:8086
```

---

## Tools Provided
### 1. `validate`
Validates the server identity for Puch AI.

**Response:**  
Returns the `MY_NUMBER` set in `.env`. 
`(The MY_NUMBER entered is in the format {COUNTRY_CODE}{PHONE_NUMBER}. For example: If your phone number is +91-1234567890 then it will return 911234567890)`

---

### 2. `web_analyzer`
Analyzes a webpage and answers questions about it.

**Expected Input:**
```json
{
  "user_query": "Summarize https://en.wikipedia.org/wiki/Apple"
}
```

**Process:**
1. Extracts URL and question from input.
2. Fetches the page HTML.
3. Cleans and converts it to Markdown.
4. Runs the analysis pipeline.
5. Returns a final answer.

---

## Notes
- The **BATCH_SIZE** and **DELAY_BETWEEN_BATCHES** variables in `web_analyzer` control request batching to Gemini to prevent rate-limit errors.
- Adjust `GEMINI_MODEL` in `.env` to use a different Gemini model.
- This implementation is **stateless**, i.e., each request is processed independently.
