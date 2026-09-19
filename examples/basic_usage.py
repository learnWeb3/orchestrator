"""End-to-end demo: gpt-5.4-mini reviewing a snippet via the `code-review` skill.

Run with: .venv/bin/python examples/basic_usage.py
Requires OPENAI_API_KEY (and optionally OPENAI_MODEL) in the environment or .env.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from orchestration_agent import Agent, Budget, StdoutLogger
from orchestration_agent.provider.openai_provider import OpenAIProvider

REPO_ROOT = Path(__file__).resolve().parent.parent

SNIPPET = '''
def get_user(request):
    user_id = request.GET["id"]
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
'''


async def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
    api_key = os.environ["OPENAI_API_KEY"]

    provider = OpenAIProvider(model=model, api_key=api_key)

    agent = Agent(
        provider=provider,
        system_prompt=(
            "You are a helpful coding assistant with access to specialized skills.\n\n"
            "{{skills_catalog}}\n\n"
            "When a task matches a skill above, call invoke_skill with that skill's "
            "name, then follow the returned instructions to answer the user."
        ),
        skills_path=str(REPO_ROOT / "skills"),
        logger=StdoutLogger(),
        budget=Budget(max_turns=5),
    )

    response = await agent.run(f"Review this code for security issues:\n\n{SNIPPET}")

    print("\n=== Response ===")
    print(response.response)
    print("\n=== Trace ===")
    for entry in response.messages:
        print(f"{entry.type}: {entry.status}")
    print("\n=== Stats ===")
    print(f"status: {response.status}")
    print(f"steps: {len(response.steps)}")
    print(f"total_tokens: {response.total_tokens}")
    print(f"token_usage: {response.token_usage}")


if __name__ == "__main__":
    asyncio.run(main())
