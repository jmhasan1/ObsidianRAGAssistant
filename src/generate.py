"""
Generation layer: takes retrieved chunks + a question, builds a grounded
RAG prompt, and calls an LLM to produce a cited answer.

Groq is the primary provider (fast, generous free tier). If GROQ_API_KEY
is missing or the call fails, falls back to OpenAI if OPENAI_API_KEY is
set. This mirrors the "prefer free tier, low-cost fallback acceptable"
constraint the assistant was scoped under.
"""

import os

SYSTEM_PROMPT = """You are a knowledge assistant answering questions about a \
personal Obsidian vault. You will be given retrieved note excerpts. \
Answer ONLY using the information in those excerpts.

Rules:
- If the excerpts don't contain enough information to answer, say so \
plainly instead of guessing.
- Cite which note(s) support each claim, using the note titles given.
- Be concise. Do not repeat the excerpts verbatim at length — synthesize.
"""


def build_prompt(question: str, hits: list) -> str:
    context_blocks = []
    for h in hits:
        context_blocks.append(
            f"[Note: {h['note_title']} | Section: {h['heading']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Retrieved context:

{context}

---

Question: {question}

Answer, citing note titles in parentheses like (Note: <title>) after each claim.
"""


def _call_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content


def generate_answer(question: str, hits: list) -> dict:
    """Returns {"answer": str, "provider": str} or raises if both fail."""
    prompt = build_prompt(question, hits)

    if os.environ.get("GROQ_API_KEY"):
        try:
            answer = _call_groq(prompt)
            return {"answer": answer, "provider": "groq/llama-3.1-8b-instant"}
        except Exception as e:
            last_err = e
    else:
        last_err = RuntimeError("GROQ_API_KEY not set")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            answer = _call_openai(prompt)
            return {"answer": answer, "provider": "openai/gpt-4o-mini"}
        except Exception as e:
            last_err = e

    raise RuntimeError(f"No LLM provider succeeded. Last error: {last_err}")
