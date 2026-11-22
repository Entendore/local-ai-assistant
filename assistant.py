"""
Omni Agent Loop: Multimodal Vision, Speculative Decoding, Parallel Tools.
"""
import json
import re
import base64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
from tools import get_tool_definitions, execute_tool

DEFAULT_SYSTEM = """\
You are an advanced local AI assistant with tool-calling capabilities.

Rules:
1. If asked about documents, files, or specific knowledge, ALWAYS use `rag_search` first.
2. You can read images natively using `analyze_image`, or extract text using `read_file`.
3. If you need to explore files, use `list_directory`.
4. If a tool requires confirmation, the user will be asked. If denied, adapt.
5. Present your final answer clearly using Markdown.
"""

def clean_and_parse_json(raw_args: str) -> dict:
    raw_args = raw_args.strip().replace("```json", "").replace("```", "").strip()
    raw_args = re.sub(r',\s*}', '}', raw_args)
    raw_args = re.sub(r',\s*]', ']', raw_args)
    return json.loads(raw_args)

class ToolCallAssistant:
    def __init__(self, model_path: str, input_callback, draft_model_path: str | None = None, system_prompt: str | None = None, max_iterations: int = 15, n_gpu_layers: int = -1, max_context_tokens: int = 7500, temperature: float = 0.7, flash_attn: bool = True, kv_cache_quant: bool = True, use_mlock: bool = True, num_threads: int = 8, n_batch: int = 512):
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.input_callback = input_callback
        
        self.tools = get_tool_definitions()
        self.tools.append({
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask the user for input or clarification if you are stuck.",
                "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
            }
        })
        
        print(f"Loading model from {model_path}...")
        kwargs = {
            "model_path": model_path, "n_gpu_layers": n_gpu_layers, "n_ctx": 8192, "n_batch": n_batch,
            "flash_attn": flash_attn, "type_k": 8 if kv_cache_quant else 0, "type_v": 8 if kv_cache_quant else 0,
            "use_mlock": use_mlock, "use_mmap": True, "num_threads": num_threads, "verbose": False
        }
        if draft_model_path:
            print(f"Loading draft model: {draft_model_path}...")
            self.draft_model = Llama(model_path=draft_model_path, n_gpu_layers=n_gpu_layers, n_ctx=8192, n_batch=n_batch, flash_attn=flash_attn, use_mlock=use_mlock, verbose=False)
            kwargs["draft_model"] = self.draft_model
        
        self.llm = Llama(**kwargs)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.token_cache = {}
        
        with open("conversation_log.md", "a", encoding="utf-8") as f:
            f.write(f"\n\n## New Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _count_tokens(self, msg: dict) -> int:
        msg_str = json.dumps(msg)
        if msg_str in self.token_cache: return self.token_cache[msg_str]
        tokens = self.llm.tokenize(str(msg).encode('utf-8'))
        self.token_cache[msg_str] = len(tokens)
        return len(tokens)

    def _trim_context(self):
        total_tokens = sum(self._count_tokens(m) for m in self.messages)
        while total_tokens > self.max_context_tokens:
            if len(self.messages) <= 3: break
            removed = self.messages.pop(1)
            total_tokens -= self._count_tokens(removed)
            while len(self.messages) > 1 and self.messages[1]["role"] == "tool":
                removed_tool = self.messages.pop(1)
                total_tokens -= self._count_tokens(removed_tool)

    def _log(self, role: str, content: str):
        with open("conversation_log.md", "a", encoding="utf-8") as f:
            if role == "user": f.write(f"### 🧑 User:\n{content}\n\n")
            elif role == "assistant": f.write(f"### 🤖 Assistant:\n{content}\n\n")
            elif role == "tool": f.write(f"**[Tool Log]** {content[:200]}...\n\n")

    def _execute_single_tool(self, tc: dict) -> dict:
        fn = tc["function"]
        name = fn["name"]
        raw_args = fn["arguments"]

        if isinstance(raw_args, str):
            try: args = clean_and_parse_json(raw_args)
            except json.JSONDecodeError:
                return {"role": "tool", "tool_call_id": tc.get("id", "null"), "name": name, "content": "Error: Malformed JSON."}
        else: args = raw_args

        if name == "ask_user": result = self.input_callback(args.get("question", "Input required:"))
        else: result = execute_tool(name, args)

        self._log("tool", f"{name} executed. Result: {result[:200]}")
        
        if isinstance(result, str) and result.startswith("VISION_IMAGE_LOADED:"):
            img_path = result.split(":", 1)[1]
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here is the image from {img_path}."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        
        return {"role": "tool", "tool_call_id": tc.get("id", "null"), "name": name, "content": result}

    def chat(self, user_input: str, stream_callback=None) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._log("user", user_input)

        for _ in range(self.max_iterations):
            self._trim_context()

            response = self.llm.create_chat_completion(
                messages=self.messages, tools=self.tools, tool_choice="auto", stream=True, max_tokens=2048, temperature=self.temperature
            )
            
            content = ""
            tool_calls = []
            
            for chunk in response:
                if not chunk["choices"]: continue
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    content += delta["content"]
                    if stream_callback: stream_callback(delta["content"])
                if "tool_calls" in delta and delta["tool_calls"]:
                    for tc in delta["tool_calls"]:
                        idx = tc["index"]
                        while len(tool_calls) <= idx:
                            tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                        if "id" in tc: tool_calls[idx]["id"] = tc["id"]
                        if "function" in tc:
                            if "name" in tc["function"]: tool_calls[idx]["function"]["name"] += tc["function"]["name"]
                            if "arguments" in tc["function"]: tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]

            msg = {"role": "assistant"}
            if content: msg["content"] = content
            if tool_calls: msg["tool_calls"] = tool_calls
            self.messages.append(msg)
            
            if not tool_calls:
                self._log("assistant", content)
                return content or "(no response)"

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(self._execute_single_tool, tool_calls))
            
            for res in results: self.messages.append(res)

        return "⚠️ Reached maximum tool-call iterations."

    def save_session(self, filepath: str):
        with open(filepath, "w") as f: json.dump(self.messages, f, indent=2)

    def load_session(self, filepath: str):
        with open(filepath, "r") as f: self.messages = json.load(f)

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
