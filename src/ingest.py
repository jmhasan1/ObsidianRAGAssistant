"""
Ingestion for an Obsidian vault.

Parses .md files (with YAML frontmatter), splits each note into
heading-based chunks, and resolves [[wikilinks]] so they can be
surfaced as related notes at query time.

Design note: chunking is heading-based rather than fixed-size because
Obsidian notes are already short and single-topic — see
vault/General/Chunking-Strategies.md for the reasoning this mirrors.
"""

import os
import re
import glob
from dataclasses import dataclass, field

import frontmatter

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


@dataclass
class Chunk:
    chunk_id: str
    note_title: str
    note_path: str
    heading: str
    text: str
    tags: list = field(default_factory=list)
    links: list = field(default_factory=list)


def _extract_links(text: str) -> list:
    return sorted(set(WIKILINK_RE.findall(text)))


def _split_by_heading(body: str):
    """Split a note body into (heading, section_text) pairs.

    Content before the first heading is kept under an empty heading
    (usually just the title, since we render the H1 as the note title).
    """
    lines = body.splitlines()
    sections = []
    current_heading = ""
    current_lines = []

    for line in lines:
        if line.startswith("#"):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    # Drop empty sections (e.g. a lone title line with no body)
    return [(h, t) for h, t in sections if t]


def load_vault(vault_dir: str) -> list:
    """Parse every .md file under vault_dir into a list of Chunk objects."""
    chunks = []
    md_files = sorted(glob.glob(os.path.join(vault_dir, "**", "*.md"), recursive=True))

    for path in md_files:
        post = frontmatter.load(path)
        tags = post.get("tags", []) or []
        body = post.content

        # Note title = first H1, falling back to filename
        title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        note_title = title_match.group(1).strip() if title_match else os.path.splitext(
            os.path.basename(path))[0]

        sections = _split_by_heading(body)
        if not sections:
            continue

        rel_path = os.path.relpath(path, vault_dir)

        for idx, (heading, text) in enumerate(sections):
            links = _extract_links(text)
            chunk = Chunk(
                chunk_id=f"{rel_path}::{idx}",
                note_title=note_title,
                note_path=rel_path,
                heading=heading or note_title,
                text=text,
                tags=tags,
                links=links,
            )
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    import sys
    vault_dir = sys.argv[1] if len(sys.argv) > 1 else "vault"
    result = load_vault(vault_dir)
    print(f"Loaded {len(result)} chunks from {vault_dir}")
    for c in result[:3]:
        print("---")
        print(c.note_title, "|", c.heading, "|", c.chunk_id)
        print(c.text[:150])
