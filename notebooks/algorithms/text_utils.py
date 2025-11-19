from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9_+#@]+")
CODE_FENCE_RE = re.compile(r"```(.*?)```", re.DOTALL)
HTML_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
INDENTED_CODE_RE = re.compile(r"(?m)(?:^|\n)(?: {4}|\t).+")

TECHNICAL_WHITELIST = {
    "int",
    "null",
    "html",
    "css",
    "sql",
    "api",
    "json",
    "ajax",
    "http",
    "https",
    "xml",
    "async",
    "await",
    "node",
    "array",
    "class",
    "object",
    "void",
    "bool",
    "byte",
    "enum",
    "char",
}


def strip_html(text: str) -> str:
    """Remove HTML tags using a lightweight regex."""
    if not isinstance(text, str):
        return ""
    return HTML_TAG_RE.sub(" ", text)


def replace_code_blocks(text: str, placeholder: str = " CODEBLOCK ") -> str:
    """Replace fenced or HTML code blocks with a placeholder token."""
    if not isinstance(text, str):
        return ""

    def _replace(pattern, input_text: str) -> str:
        return pattern.sub(placeholder, input_text)

    text = _replace(CODE_FENCE_RE, text)
    text = _replace(HTML_CODE_RE, text)
    return text


def normalize_text(text: str) -> str:
    """Lower-case text and collapse whitespace."""
    text = text or ""
    text = text.lower()
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokenize using a simple regex tokenizer that keeps technical tokens intact."""
    return TOKEN_RE.findall(text)


def filter_stopwords(tokens: Sequence[str], stopwords: Iterable[str]) -> List[str]:
    stopword_set = set(stopwords) - TECHNICAL_WHITELIST
    return [tok for tok in tokens if tok and tok not in stopword_set]


def stem_tokens(tokens: Sequence[str], stemmer) -> List[str]:
    if stemmer is None:
        return list(tokens)
    return [stemmer.stem(tok) for tok in tokens]


def preprocess_text(
    text: str,
    stopwords: Iterable[str],
    stemmer=None,
) -> Tuple[str, List[str]]:
    """Return normalized text and token list."""
    text = strip_html(text)
    text = replace_code_blocks(text)
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    tokens = filter_stopwords(tokens, stopwords)
    tokens = stem_tokens(tokens, stemmer)
    return normalized, tokens


def tokens_to_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def extract_code_statistics(text: str) -> Tuple[int, int]:
    """Return (num_blocks, num_lines) heuristically detected from code fences and HTML tags."""
    if not isinstance(text, str):
        return 0, 0
    blocks = []
    for pattern in (CODE_FENCE_RE, HTML_CODE_RE):
        blocks.extend(pattern.findall(text))
    indented = INDENTED_CODE_RE.findall(text)
    if indented:
        blocks.append("\n".join(indented))
    num_blocks = len(blocks)
    num_lines = sum(block.count("\n") + 1 for block in blocks)
    return num_blocks, num_lines
