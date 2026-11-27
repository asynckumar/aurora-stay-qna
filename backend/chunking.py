"""
Structure-aware chunker for the AuroraStay policy documents.

Design (driven by measurements, see the notebook's chunking section):

1. The embedding model (HarishMaths/Hotel-Policy-Embedding) reads at most
   128 tokens. Anything longer is silently truncated before it is embedded,
   so every chunk is kept under that window.
2. Docling exports tables as space-padded markdown, which wastes most of the
   token budget on padding and pipes. Each table row is rewritten as one
   compact "record" that carries its column names, e.g.
   "Property: ASH-MIA Brickell, Miami FL; Parking: Valet, $55 per night; ..."
   A row is self-contained, so it can be retrieved on its own.
3. Every policy is organized as numbered sections. Docling turns boxed
   call-outs ("Guest-facing summary", "Escalate immediately") into their own
   headings; they are merged back into the numbered section they belong to.
4. Every chunk starts with "Document title | Section N, heading" so both the
   embedding and the LLM know where the text came from, and so answers can
   cite the exact section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

NUMBERED = re.compile(r"^(\d+)\.\s+(.*)$")
TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _split_sections(markdown: str):
    """Yield (heading, body) for every '## ' heading in a Docling export."""
    for part in re.split(r"^## +", markdown, flags=re.M):
        if part.strip():
            heading, _, body = part.partition("\n")
            yield heading.strip(), body.strip()


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _blocks(body: str):
    """Split a section body into ('para', text) and ('row', text) blocks.

    Table rows are converted to 'Column: value; Column: value' records.
    """
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and i + 1 < len(lines) and TABLE_SEP.match(lines[i + 1].strip()):
            header = _cells(line)
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                vals = _cells(lines[i])
                pairs = [f"{h}: {v}" for h, v in zip(header, vals) if v]
                yield "row", "; ".join(pairs) + "."
                i += 1
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("|"):
            para.append(lines[i].strip())
            i += 1
        if para:
            yield "para", " ".join(para)
        else:
            i += 1


def _group_sections(markdown: str):
    """Group headings into numbered sections (call-outs join their section)."""
    sections = list(_split_sections(markdown))
    title = sections[0][0] if sections else ""
    groups: list[dict] = []
    for heading, body in sections:
        num = NUMBERED.match(heading)
        if num:
            groups.append({"number": int(num.group(1)),
                           "section": f"Section {num.group(1)}, {num.group(2)}",
                           "parts": [body]})
        elif not groups:
            groups.append({"number": 0, "section": "Overview", "parts": [body]})
        elif groups[-1]["number"] == 0:
            if heading.lower() == "document map":
                continue  # identical navigation table in every document
            groups[-1]["parts"].append(f"{heading}: {body}")
        else:
            # Call-out box: keep its label so the meaning survives.
            groups[-1]["parts"].append(f"{heading}: {body}")
    return title, groups


def structure_chunks(markdown: str, source: str, tokenizer=None,
                     max_tokens: int = 120, include_overview: bool = True) -> list[Chunk]:
    """Turn one Docling markdown export into compact, self-contained chunks.

    Paragraphs and table rows are packed together in document order until
    the next block would push the chunk past ``max_tokens``. A single block
    that is longer than the budget is split on sentence boundaries.
    """
    title, groups = _group_sections(markdown)
    code = re.search(r"Policy code\s*\|\s*([A-Z]{2}-\d+)", markdown)
    policy_code = code.group(1) if code else ""

    def n_tokens(text: str) -> int:
        if tokenizer is None:
            return int(len(text.split()) * 1.35)
        return len(tokenizer.tokenize(text))

    chunks: list[Chunk] = []
    for g in groups:
        if g["number"] == 0 and not include_overview:
            continue
        header = f"{title} | {g['section']}"
        blocks: list[tuple[str, str]] = []
        for part in g["parts"]:
            blocks.extend(_blocks(part))

        # Split any single block that is longer than the budget.
        budget = max_tokens - n_tokens(header) - 2
        units: list[tuple[str, str]] = []
        for kind, text in blocks:
            if n_tokens(text) <= budget:
                units.append((kind, text))
                continue
            buf = ""
            for sent in re.split(r"(?<=[.;])\s+", text):
                cand = f"{buf} {sent}".strip()
                if buf and n_tokens(cand) > budget:
                    units.append((kind, buf))
                    buf = sent
                else:
                    buf = cand
            if buf:
                units.append((kind, buf))

        # Pack consecutive units of the same kind (rows with rows,
        # paragraphs with paragraphs) so a table row is never glued to an
        # unrelated paragraph.
        packed: list[tuple[str, list[str]]] = []
        for kind, text in units:
            if packed and packed[-1][0] == kind and \
                    n_tokens(header + " " + " ".join(packed[-1][1] + [text])) <= max_tokens and kind == "para":
                packed[-1][1].append(text)
            else:
                packed.append((kind, [text]))

        for idx, (kind, texts) in enumerate(packed):
            body = "\n".join(texts)
            chunks.append(Chunk(
                text=f"{header}\n{body}",
                metadata={
                    "source": source,
                    "document": title,
                    "policy_code": policy_code,
                    "section": g["section"],
                    "section_number": g["number"],
                    "block_type": "table_row" if kind == "row" else "text",
                    "position": idx,
                },
            ))
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['policy_code'] or 'DOC'}-{c.metadata['section_number']}-{c.metadata['position']}"
    return chunks


def property_aliases(markdown_docs: list[str]) -> dict[str, str]:
    """Build a city/name -> property-code map from the property matrix rows.

    'ASH-MIA Brickell, Miami FL' gives {'miami': 'ASH-MIA Brickell',
    'brickell': 'ASH-MIA Brickell', 'mia': 'ASH-MIA Brickell'}. Built from
    the documents themselves, so a new property row extends it for free.
    """
    aliases: dict[str, str] = {}
    pat = re.compile(r"\|\s*(ASH-([A-Z]{3})\s+([A-Za-z ]+?))(?:,\s*([A-Za-z ]+?)\s+[A-Z]{2})?\s*\|")
    for md in markdown_docs:
        for m in pat.finditer(md):
            name, code3, area, city = m.group(1).strip(), m.group(2), m.group(3), m.group(4)
            for key in filter(None, [code3, area, city]):
                aliases.setdefault(key.strip().lower(), name)
    return aliases
