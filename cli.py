#!/usr/bin/env python3
"""
CLI entry point for the CV QA Agent.

Run with:
    python cli.py

Commands:
    exit / quit  -- end the session (saves output file)
    clear        -- clear conversation history
    help         -- show available commands
"""

from __future__ import annotations

import io
import sys

# Force UTF-8 stdout on Windows so Rich/emoji don't crash
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Rich for pretty terminal output ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.rule import Rule

    _rich = True
    # Force UTF-8 console so box-drawing chars work on Windows PowerShell
    console = Console(
        force_terminal=True,
        highlight=False,
        file=sys.stdout,
    )
except ImportError:
    _rich = False
    console = None  # type: ignore[assignment]


def _print(msg: str, style: str = "") -> None:
    if _rich:
        console.print(msg, style=style)
    else:
        print(msg)


def _print_panel(content: str, title: str = "", style: str = "blue") -> None:
    if _rich:
        console.print(Panel(content, title=title, border_style=style))
    else:
        print(f"\n--- {title} ---\n{content}\n")


BANNER = """\
+============================================================+
|          CV Question Answering Agent                       |
|          Powered by LangGraph + Google Gemini              |
+============================================================+
  Type your question and press Enter.
  Commands: exit | quit | clear | help
"""

HELP_TEXT = """\
Available commands:
  exit / quit  -- End the session and save output file
  clear        -- Clear conversation history (start fresh)
  help         -- Show this help message
  <question>   -- Ask anything about the CV
"""


def main() -> None:
    # ── Startup ───────────────────────────────────────────────────────────────
    print(BANNER)
    print("Initialising agent... (this may take a moment on first run)\n")

    try:
        from cv_agent import CVAgent
    except Exception as exc:
        print(f"\n[ERROR] Failed to import CVAgent: {exc}")
        sys.exit(1)

    try:
        agent = CVAgent()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("Please place your CV file at the path shown above and re-run.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Startup failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if _rich:
        console.print(Rule(style="green"))
        console.print("[bold green]Agent ready! Start asking questions.[/]")
        console.print(Rule(style="green"))
    else:
        print("\nAgent ready!\n" + "=" * 60)

    # ── REPL loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            if _rich:
                user_input = Prompt.ask("\n[bold blue]You[/]").strip()
            else:
                user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted. Saving session...")
            agent.save_session()
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Commands ──────────────────────────────────────────────────────────
        if cmd in {"exit", "quit"}:
            print("\nGoodbye! Saving session...")
            agent.save_session()
            break

        if cmd == "clear":
            agent.clear_history()
            _print("Conversation history cleared.", style="yellow")
            continue

        if cmd == "help":
            _print_panel(HELP_TEXT, title="Help", style="cyan")
            continue

        # ── Ask the agent ─────────────────────────────────────────────────────
        if _rich:
            with console.status("[bold yellow]Thinking...[/]", spinner="dots"):
                answer = agent.ask(user_input)
        else:
            print("Thinking...")
            answer = agent.ask(user_input)

        if _rich:
            console.print(Rule(style="dim"))
            console.print(Panel(
                Markdown(answer),
                title=f"[bold green]Agent[/] (turn {agent.turn_count})",
                border_style="green",
            ))
        else:
            print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
