"""Claude LLM integration for answer generation."""

import anthropic


def generate_answer(
    query: str,
    context_chunks: list[dict],
    api_key: str | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Generate an answer using Claude, given a query and retrieved context.

    Args:
        query: The user's question.
        context_chunks: List of retrieved chunk dicts (each with 'text' key).
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model: Claude model to use. Defaults to claude-haiku-4-5-20251001.

    Returns:
        The generated answer string.
    """
    context = "\n\n---\n\n".join(chunk["text"] for chunk in context_chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions based on the "
        "provided context. Use only the context below to answer. If the "
        "context does not contain enough information, say so clearly."
    )

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer based on the context above:"
    )

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
