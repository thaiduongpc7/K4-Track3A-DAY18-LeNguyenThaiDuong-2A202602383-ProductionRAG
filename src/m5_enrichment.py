from __future__ import annotations

"""Module 5: LLM-assisted chunk enrichment with offline fallbacks."""

import json
import os
import re
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """A chunk together with generated enrichment fields."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def summarize_chunk(text: str) -> str:
    """Create a short summary using GPT when available, otherwise extractively."""
    text = str(text or "").strip()
    if not text:
        return ""

    if OPENAI_API_KEY:
        try:
            content = _chat_completion(
                system_prompt=(
                    "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
                    "Chỉ trả về phần tóm tắt."
                ),
                user_prompt=text,
                max_tokens=150,
            )
            if content:
                return content
        except Exception as exc:  # noqa: BLE001 - enrichment must have a fallback
            print(f"  Warning: OpenAI summarize failed: {exc}")

    return _extractive_summary(text)


def generate_hypothesis_questions(
    text: str, n_questions: int = 3
) -> list[str]:
    """Generate likely user questions that the chunk can answer."""
    if n_questions <= 0:
        return []
    text = str(text or "").strip()
    if not text:
        return []

    if OPENAI_API_KEY:
        try:
            content = _chat_completion(
                system_prompt=(
                    f"Dựa trên đoạn văn, tạo tối đa {n_questions} câu hỏi mà "
                    "đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên một dòng."
                ),
                user_prompt=text,
                max_tokens=200,
            )
            questions = _parse_questions(content, n_questions)
            if questions:
                return questions
        except Exception as exc:  # noqa: BLE001 - enrichment must have a fallback
            print(f"  Warning: OpenAI HyQA failed: {exc}")

    return _fallback_questions(text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a short document/topic description while preserving the chunk."""
    text = str(text or "").strip()
    if not text:
        return ""

    if OPENAI_API_KEY:
        try:
            content = _chat_completion(
                system_prompt=(
                    "Viết một câu ngắn bằng tiếng Việt mô tả đoạn văn nằm ở đâu "
                    "trong tài liệu và nói về chủ đề gì. Chỉ trả về một câu."
                ),
                user_prompt=(
                    f"Tài liệu: {document_title or '(không rõ tiêu đề)'}\n\n"
                    f"Đoạn văn:\n{text}"
                ),
                max_tokens=80,
            )
            if content:
                return f"{content}\n\n{text}"
        except Exception as exc:  # noqa: BLE001 - enrichment must have a fallback
            print(f"  Warning: OpenAI contextual enrichment failed: {exc}")

    prefix = f"Trích từ tài liệu {document_title}. " if document_title else "Đoạn trích tài liệu. "
    return f"{prefix}\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract useful metadata with GPT or infer conservative local metadata."""
    text = str(text or "").strip()
    if OPENAI_API_KEY and text:
        try:
            content = _chat_completion(
                system_prompt=(
                    'Trích xuất metadata và chỉ trả về JSON hợp lệ với schema: '
                    '{"topic": "...", "entities": ["..."], '
                    '"category": "policy|hr|it|finance|general", '
                    '"language": "vi|en"}'
                ),
                user_prompt=text,
                max_tokens=150,
            )
            metadata = _parse_json_object(content)
            if metadata:
                return _normalise_metadata(metadata, text)
        except Exception as exc:  # noqa: BLE001 - enrichment must have a fallback
            print(f"  Warning: OpenAI metadata extraction failed: {exc}")

    return _fallback_metadata(text)


def _enrich_single_call(text: str, source: str) -> dict:
    """Return summary, HyQA, context and metadata from one LLM call."""
    text = str(text or "").strip()
    source = str(source or "").strip()
    if OPENAI_API_KEY and text:
        try:
            content = _chat_completion(
                system_prompt=(
                    "Phân tích đoạn văn và trả về JSON hợp lệ, không thêm markdown, "
                    "theo đúng schema:\n"
                    "{\n"
                    '  "summary": "tóm tắt 2-3 câu",\n'
                    '  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],\n'
                    '  "context": "một câu mô tả vị trí và chủ đề của đoạn văn",\n'
                    '  "metadata": {"topic": "...", "entities": [], '
                    '"category": "policy|hr|it|finance|general", "language": "vi|en"}\n'
                    "}"
                ),
                user_prompt=f"Tài liệu: {source or '(không rõ tiêu đề)'}\n\nĐoạn văn:\n{text}",
                max_tokens=400,
            )
            parsed = _parse_json_object(content)
            if parsed:
                return _normalise_enrichment(parsed, text, source)
        except Exception as exc:  # noqa: BLE001 - enrichment must have a fallback
            print(f"  Warning: combined enrichment failed: {exc}")

    return _fallback_enrichment(text, source)


def _chat_completion(
    system_prompt: str, user_prompt: str, max_tokens: int
) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return str(content or "").strip()


def _extractive_summary(text: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return text
    summary = " ".join(sentences[:2])
    return summary if summary.endswith((".", "!", "?")) else f"{summary}."


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    sentences = _sentences(text)
    questions: list[str] = []

    number_match = re.search(
        r"\b\d+(?:[.,]\d+)?\s*(?:ngày|tháng|năm|giờ|%|đồng|người)\b",
        text.casefold(),
    )
    if number_match:
        questions.append(
            f"Đoạn văn quy định bao nhiêu {number_match.group(0).split(maxsplit=1)[1]}?"
        )

    if sentences:
        topic = _trim_question_source(sentences[0])
        questions.append(f"Đoạn văn này nói về điều gì liên quan đến {topic}?")

    for sentence in sentences[1:]:
        questions.append(f"Quy định cụ thể trong đoạn văn là gì: {sentence}?")

    deduplicated = []
    for question in questions:
        question = re.sub(r"\s+", " ", question).strip()
        if question not in deduplicated:
            deduplicated.append(question)
    return deduplicated[:n_questions]


def _fallback_metadata(text: str) -> dict:
    lowered = text.casefold()
    if any(keyword in lowered for keyword in ("mật khẩu", "vpn", "phần mềm", "công nghệ")):
        category = "it"
    elif any(keyword in lowered for keyword in ("lương", "nghỉ phép", "nhân viên", "thử việc")):
        category = "hr"
    elif any(keyword in lowered for keyword in ("doanh thu", "chi phí", "tài chính", "cổ phiếu")):
        category = "finance"
    elif any(keyword in lowered for keyword in ("quy định", "chính sách", "điều khoản")):
        category = "policy"
    else:
        category = "general"

    entities = re.findall(r"\b[A-ZĐ][\wĐ]*(?:\s+[A-ZĐ][\wĐ]*)*\b", text)
    return {
        "topic": _trim_question_source(_sentences(text)[0]) if _sentences(text) else "general",
        "entities": list(dict.fromkeys(entities))[:10],
        "category": category,
        "language": "vi" if _looks_vietnamese(text) else "en",
    }


def _fallback_enrichment(text: str, source: str) -> dict:
    return {
        "summary": _extractive_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": (
            f"Đoạn trích thuộc tài liệu {source}."
            if source
            else "Đoạn trích thuộc tài liệu hiện tại."
        ),
        "metadata": _fallback_metadata(text),
    }


def _normalise_enrichment(value: dict, text: str, source: str) -> dict:
    fallback = _fallback_enrichment(text, source)
    questions = value.get("questions", fallback["questions"])
    if isinstance(questions, str):
        questions = _parse_questions(questions, 3)
    elif isinstance(questions, list):
        questions = [str(question).strip() for question in questions if str(question).strip()]
    else:
        questions = fallback["questions"]

    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        metadata = fallback["metadata"]
    else:
        metadata = _normalise_metadata(metadata, text)

    return {
        "summary": str(value.get("summary") or fallback["summary"]).strip(),
        "questions": questions[:3],
        "context": str(value.get("context") or fallback["context"]).strip(),
        "metadata": metadata,
    }


def _normalise_metadata(value: dict, text: str) -> dict:
    fallback = _fallback_metadata(text)
    entities = value.get("entities", fallback["entities"])
    if isinstance(entities, str):
        entities = [entities]
    if not isinstance(entities, list):
        entities = fallback["entities"]
    return {
        **fallback,
        **value,
        "topic": str(value.get("topic") or fallback["topic"]),
        "entities": [str(entity) for entity in entities],
        "category": str(value.get("category") or fallback["category"]),
        "language": str(value.get("language") or fallback["language"]),
    }


def _parse_json_object(content: str) -> dict:
    if not content:
        return {}
    candidate = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(candidate[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def _parse_questions(content: str, limit: int) -> list[str]:
    questions = []
    for line in str(content or "").splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            questions.append(cleaned if cleaned.endswith("?") else f"{cleaned}?")
    return questions[:limit]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def _trim_question_source(sentence: str) -> str:
    words = sentence.rstrip(".!?").split()
    return " ".join(words[:12]).strip()


def _looks_vietnamese(text: str) -> bool:
    return bool(re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]", text))


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Run combined or individual enrichment methods for every chunk."""
    if methods is None:
        methods = ["combined"]
    if not methods:
        return []

    use_combined = "combined" in methods
    enriched = []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text", ""))
        metadata = dict(chunk.get("metadata") or {})
        source = str(metadata.get("source", ""))

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result["summary"]
            questions = result["questions"]
            context_line = result["context"]
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result["metadata"]
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = (
                generate_hypothesis_questions(text)
                if "hyqa" in methods
                else []
            )
            enriched_text = (
                contextual_prepend(text, source)
                if "contextual" in methods
                else text
            )
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**metadata, **auto_meta},
                method="+".join(methods),
            )
        )

        if (index + 1) % 10 == 0 or (index + 1) == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)
    return enriched


if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )
    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary: {summarize_chunk(sample)}\n")
    print(f"HyQA questions: {generate_hypothesis_questions(sample)}\n")
    print(f"Contextual: {contextual_prepend(sample, 'Sổ tay nhân viên VinUni 2024')}\n")
    print(f"Auto metadata: {extract_metadata(sample)}")
