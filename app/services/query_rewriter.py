"""
Query Rewriter — rewrites vague/follow-up questions into keyword-rich
search queries before embedding, improving retrieval quality.

Now uses Groq instead of local Ollama for inference.
Still falls back to the original question silently on any failure.
"""
from groq import Groq
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

REWRITE_PROMPT = """\
You are a search query optimizer for a document retrieval system.

Your job: rewrite the user's question into a concise, keyword-rich search query \
that will retrieve the most relevant passages from a PDF document.

Rules:
- Output ONLY the rewritten query — no explanation, no preamble, no quotes
- Keep it under 20 words
- Replace pronouns (he, she, it, they, that, this) with the actual subject from the conversation
- For follow-up questions, make them self-contained using the conversation history
- Preserve technical terms, names, numbers, and dates exactly
- If the question is already a good search query, return it unchanged

Conversation history (last few turns):
{history}

Current question: {question}

Rewritten query:"""


class QueryRewriter:
    def __init__(self):
        settings = get_settings()
        self._enabled = settings.query_rewrite_enabled
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    def rewrite(self, question: str, history_text: str = "") -> tuple[str, bool]:
        """
        Rewrite the question for better retrieval.
        Returns (rewritten_query, was_rewritten).
        Falls back to original question on any failure.
        """
        if not self._enabled:
            return question, False

        # Skip rewriting for short direct queries
        if len(question.split()) <= 6 and "?" not in question:
            return question, False

        # Skip if no history and it's already a clean query
        if not history_text.strip() and self._looks_like_good_query(question):
            return question, False

        try:
            prompt = REWRITE_PROMPT.format(
                history=history_text.strip() or "None",
                question=question,
            )

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )

            rewritten = response.choices[0].message.content.strip().strip('"\'')

            # Sanity checks
            if not rewritten:
                raise ValueError("Empty rewrite")
            if len(rewritten) > 300:
                raise ValueError("Rewrite too long")
            if rewritten.lower().startswith(("i ", "sure", "here", "of course", "the rewritten")):
                raise ValueError("LLM ignored instructions")

            logger.info("query_rewritten",
                        original=question[:60], rewritten=rewritten[:60])
            return rewritten, True

        except Exception as e:
            logger.warning("query_rewrite_failed", error=str(e), question=question[:60])
            return question, False

    def _looks_like_good_query(self, question: str) -> bool:
        vague_starters = (
            "tell me more", "what about", "explain that", "elaborate",
            "what did", "what does", "how about", "and what", "also",
            "can you", "could you", "please", "what else",
        )
        question_lower = question.lower().strip()
        if any(question_lower.startswith(v) for v in vague_starters):
            return False
        pronoun_signals = {" it ", " that ", " this ", " they ", " he ", " she ",
                           " him ", " her ", " them ", " those ", " these "}
        padded = f" {question_lower} "
        if any(p in padded for p in pronoun_signals):
            return False
        return True


_rewriter: QueryRewriter | None = None

def get_query_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter