"""Terminal chat loop for the agentic resume screening system.

Thin CLI alternative to app.py. All agent logic lives in matching_agent
(build_graph / run_turn) — this file only drives a prompt loop and prints
the response plus the node/tool trace, so there's no duplicated agent code
between the two interfaces.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from matching_agent import build_graph, run_turn


def main() -> None:
    app = build_graph()
    config = {"configurable": {"thread_id": "cli-session"}}

    print("Agentic Profile Matching — CLI")
    print('Type a job description, a JD filepath, or a follow-up instruction. Type "exit" to quit.\n')

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        turn = run_turn(app, config, user_input)
        messages = turn["state"]["messages"]
        last_human_idx = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        turn_ai_messages = [m for m in messages[last_human_idx + 1:] if isinstance(m, AIMessage)]

        print()
        for message in turn_ai_messages:
            print(message.content)
            print()
        if not turn_ai_messages:
            print("(No report generated this turn — see the trace below.)\n")

        print("--- trace ---")
        for step in turn["trace"]:
            tools = ", ".join(step["tools_called"]) if step["tools_called"] else "-"
            print(f"  {step['node']:22s} tools: {tools}")
        print()


if __name__ == "__main__":
    main()
