"""
Interactive CLI with Live Markdown & Multimodal Support.
"""
import sys
import os
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Confirm
from assistant import ToolCallAssistant
from rag import LocalRAG
import tools

console = Console()

BANNER = """
[bold cyan]
  ╔═══════════════════════════════════════════════╗
  ║   Local AI Assistant — llama.cpp v10.0 (Omni) ║
  ╚═══════════════════════════════════════════════╝
[/bold cyan]
"""

HELP = """
[bold]Commands:[/bold]
  /help           Show this help
  /tools          List available tools
  /index <file>   Manually index a file into RAG
  /list_rag       List indexed files
  /rag_clear      Clear the RAG database
  /clear          Reset conversation
  /save <file>    Save session
  /load <file>    Load session
  /exit           Quit
"""

def get_user_input(prompt_text):
    console.print(f"\n[bold yellow]🤖 Assistant needs input:[/bold yellow] {prompt_text}")
    return console.input("[bold blue]🧑 You:[/bold blue] ")

def confirm_action(tool_name: str, args: dict) -> bool:
    console.print("\n[bold red]⚠️ Action Required![/bold red]")
    console.print(f"The assistant wants to use: [bold yellow]{tool_name}[/bold yellow]")
    console.print(f"Arguments: [cyan]{json.dumps(args, indent=2)}[/cyan]")
    return Confirm.ask("[bold red]Do you approve this action?[/bold red]", default=False)

def main():
    console.print(BANNER)
    
    config = {
        "model_path": None, "draft_model_path": None, "embed_model_path": None,
        "system_prompt": None, "temperature": 0.7, "n_gpu_layers": -1,
        "flash_attn": True, "kv_cache_quant": True, "use_mlock": True,
        "num_threads": 8, "n_batch": 512
    }
    if os.path.exists("config.json"):
        with open("config.json", "r") as f: config.update(json.load(f))
    
    embed_path = config.get("embed_model_path")
    if embed_path and os.path.exists(embed_path):
        try:
            rag = LocalRAG(
                embed_model_path=embed_path, 
                n_gpu_layers=config["n_gpu_layers"],
                flash_attn=config["flash_attn"],
                num_threads=config["num_threads"]
            )
            tools.set_rag_instance(rag)
        except Exception as e:
            console.print(f"[red]Failed to load RAG model: {e}[/red]")
    else:
        console.print("[yellow]No embed_model_path in config. RAG tools disabled.[/yellow]")
    
    tools.set_confirm_callback(confirm_action)
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else config.get("model_path")
    if not model_path:
        ggufs = [f for f in os.listdir(".") if f.endswith(".gguf") and "embed" not in f.lower() and "bge" not in f.lower()]
        if ggufs:
            model_path = ggufs[0]
            console.print(f"  No model specified. Found and using: [yellow]{model_path}[/yellow]")
        else:
            console.print("  [red]Error: No LLM model specified.[/red]")
            return

    console.print(f"  Model: [green]{model_path}[/green]\n")

    try:
        assistant = ToolCallAssistant(
            model_path=model_path,
            input_callback=get_user_input,
            draft_model_path=config.get("draft_model_path"),
            system_prompt=config.get("system_prompt"),
            temperature=config.get("temperature", 0.7),
            n_gpu_layers=config.get("n_gpu_layers", -1),
            flash_attn=config.get("flash_attn", True),
            kv_cache_quant=config.get("kv_cache_quant", True),
            use_mlock=config.get("use_mlock", True),
            num_threads=config.get("num_threads", 8),
            n_batch=config.get("n_batch", 512)
        )
    except Exception as e:
        console.print(f"  [red]Failed to load model: {e}[/red]")
        return

    while True:
        try:
            user_input = console.input("\n[bold blue]🧑 You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [yellow]Bye! 👋[/yellow]")
            break

        if not user_input: continue

        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            args = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd == "/exit": break
            elif cmd == "/help": console.print(Panel(HELP, title="Help"))
            elif cmd == "/clear":
                assistant.reset()
                console.print("  [green]✅ Conversation cleared.[/green]")
            elif cmd == "/save":
                if not args: console.print("  [red]Usage: /save <filename>[/red]"); continue
                assistant.save_session(args)
                console.print(f"  [green]✅ Session saved to {args}[/green]")
            elif cmd == "/load":
                if not args: console.print("  [red]Usage: /load <filename>[/red]"); continue
                if os.path.exists(args):
                    assistant.load_session(args)
                    console.print(f"  [green]✅ Session loaded[/green]")
            elif cmd == "/index":
                if not args: console.print("  [red]Usage: /index <filepath>[/red]"); continue
                if tools.RAG_INSTANCE:
                    with console.status(f"Indexing [cyan]{args}[/cyan]..."):
                        msg = tools.RAG_INSTANCE.add_text(tools.extract_text_from_file(args), args)
                    console.print(f"  [green]✅ {msg}[/green]")
            elif cmd == "/list_rag":
                if tools.RAG_INSTANCE:
                    console.print(Panel(tools.RAG_INSTANCE.list_files(), title="Indexed Files"))
            elif cmd == "/rag_clear":
                if tools.RAG_INSTANCE:
                    tools.RAG_INSTANCE.clear_db()
                    console.print("  [green]✅ RAG database cleared.[/green]")
            elif cmd == "/tools":
                tool_list = [f"• [bold yellow]{t['function']['name']}[/bold yellow]: {t['function']['description']}" for t in tools.get_tool_definitions()]
                console.print(Panel("\n".join(tool_list), title="Available Tools"))
            else:
                console.print(f"  [red]Unknown command: {user_input}[/red]")
            continue

        console.print("\n[bold magenta]🤖 Assistant:[/bold magenta]")
        
        full_response = ""
        with Live(Markdown(""), console=console, refresh_per_second=10, vertical_overflow="visible") as live:
            def stream_handler(text_chunk):
                nonlocal full_response
                full_response += text_chunk
                live.update(Markdown(full_response))
            
            try:
                assistant.chat(user_input, stream_callback=stream_handler)
            except Exception as e:
                console.print(f"\n  [red]❌ Error during chat: {e}[/red]")

if __name__ == "__main__":
    main()
