import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from langchain_openai import ChatOpenAI


def structured_llm(schema, *, temperature):
    """Shared model configuration for proposer and critic."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    options = {"seed": int(os.environ["OPENAI_SEED"])} if "OPENAI_SEED" in os.environ else {}
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                      temperature=temperature, **options).with_structured_output(
                          schema, method="json_schema", strict=True)
