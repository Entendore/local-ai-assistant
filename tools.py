"""
Tool registry with Vision, OCR, RAG, Memory, and Charts.
"""
import os
import json
import subprocess
import tempfile
import requests
from datetime import datetime
from typing import Callable

RAG_INSTANCE = None
CONFIRM_CALLBACK = None

def set_rag_instance(rag):
    global RAG_INSTANCE
    RAG_INSTANCE = rag

def set_confirm_callback(cb):
    global CONFIRM_CALLBACK
    CONFIRM_CALLBACK = cb

_TOOLS: dict[str, dict] = {}

def tool(name: str, description: str, parameters: dict):
    def decorator(func: Callable):
        func.requires_confirmation = getattr(func, "requires_confirmation", False)
        _TOOLS[name] = {
            "function": func,
            "definition": {
                "type": "function",
                "function": {
                    "name": name, "description": description, "parameters": parameters,
                },
            },
        }
        return func
    return decorator

def requires_confirmation(func: Callable):
    func.requires_confirmation = True
    return func

def get_tool_definitions() -> list[dict]:
    return [t["definition"] for t in _TOOLS.values()]

def execute_tool(name: str, arguments: dict) -> str:
    if name not in _TOOLS: return f"Error: Unknown tool '{name}'"
    func = _TOOLS[name]["function"]
    if getattr(func, "requires_confirmation", False) and CONFIRM_CALLBACK:
        if not CONFIRM_CALLBACK(name, arguments): return "Action cancelled by user."
    try:
        return str(func(**arguments))
    except Exception as e:
        return f"Error executing '{name}': {e}"

# ─── File Extraction & Vision ───────────────────────────────

def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(filepath))
        except Exception: return ""
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if len(text.strip()) < 20:
                from pdf2image import convert_from_path
                pages = convert_from_path(filepath)
                return "\n".join(pytesseract.image_to_string(p) for p in pages)
            return text
        except Exception: return ""
    elif ext == ".pptx":
        try:
            from pptx import Presentation
            prs = Presentation(filepath)
            return "\n".join(s.text for slide in prs.slides for s in slide.shapes if hasattr(s, "text") and s.text.strip())
        except Exception: return ""
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f: return f.read()

# ─── Tools ──────────────────────────────────────────────────

@tool(
    name="analyze_image",
    description="Analyze an image file natively using Vision capabilities. Use this instead of read_file if you need to understand the visual content of a photo, diagram, or screenshot.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)
def analyze_image(path: str) -> str:
    if not os.path.exists(path): return "Error: File not found."
    return f"VISION_IMAGE_LOADED:{path}"

@tool(name="rag_index_file", description="Index a file into the local RAG knowledge base.",
      parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
def rag_index_file(path: str) -> str:
    if not RAG_INSTANCE: return "RAG not initialized."
    text = extract_text_from_file(path)
    return RAG_INSTANCE.add_text(text, source=path) if text.strip() else "Error: No text extracted."

@tool(name="rag_search", description="Search the local RAG knowledge base.",
      parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
def rag_search(query: str) -> str:
    if not RAG_INSTANCE: return "RAG not initialized."
    return RAG_INSTANCE.search(query)

@tool(name="save_memory", description="Save a fact to long-term memory.",
      parameters={"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]})
def save_memory(key: str, value: str) -> str:
    memories = {}
    if os.path.exists("memory.json"):
        with open("memory.json", "r") as f: memories = json.load(f)
    memories[key] = value
    with open("memory.json", "w") as f: json.dump(memories, f, indent=2)
    return f"Saved to memory: {key} = {value}"

@tool(name="read_file", description="Read text, PDF, or PPTX files.",
      parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
def read_file(path: str) -> str:
    return extract_text_from_file(path)[:8000]

@tool(name="list_directory", description="List files in a directory.",
      parameters={"type": "object", "properties": {"path": {"type": "string", "default": "."}}})
def list_directory(path: str = ".") -> str:
    if not os.path.exists(path): return f"Error: Directory '{path}' not found."
    out = [f"{'[DIR]' if os.path.isdir(os.path.join(path,i)) else '[FILE]'} {i}" for i in os.listdir(path)]
    return "\n".join(out) if out else "(empty)"

@requires_confirmation
@tool(name="write_file", description="Write content to a file.",
      parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]})
def write_file(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as f: f.write(content)
    return f"Wrote {len(content)} chars to {path}."

@requires_confirmation
@tool(name="python_repl", description="Execute Python 3 code.",
      parameters={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]})
def python_repl(code: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code); temp_path = f.name
    try:
        res = subprocess.run(["python", temp_path], capture_output=True, text=True, timeout=30)
        return (res.stdout + (f"\n[STDERR]\n{res.stderr}" if res.stderr else ""))[:5000] or "(no output)"
    finally:
        os.remove(temp_path)

@requires_confirmation
@tool(name="generate_chart", description="Generate a matplotlib chart from Python code and save as PNG. The code should create a plot and NOT call plt.show().",
      parameters={"type": "object", "properties": {"code": {"type": "string"}, "filename": {"type": "string"}}, "required": ["code", "filename"]})
def generate_chart(code: str, filename: str) -> str:
    full_code = f"import matplotlib\nmatplotlib.use('Agg')\n{code}\nplt.savefig('{filename}', dpi=150)\nprint('Chart saved.')"
    return python_repl(full_code)

@tool(name="calculate", description="Evaluate math safely.",
      parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})
def calculate(expression: str) -> str:
    import math
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max})
    return f"{expression} = {eval(expression, {'__builtins__': {}}, allowed)}"

@tool(name="web_search", description="Search the web using DuckDuckGo.",
      parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
def web_search(query: str) -> str:
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = [f"Title: {r.find('a', class_='result__a').get_text()}\nSnippet: {r.find('a', class_='result__snippet').get_text()}" for r in soup.find_all('div', class_='result')[:5]]
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e: return f"Search failed: {e}"
