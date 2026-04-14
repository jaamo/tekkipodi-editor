from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path

from rich.console import Console

from .io_utils import require_tone_of_voice
from .models import Session, TranscriptChunk
from .transcribe import transcribe_session

_console = Console()

DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 8192
SLUG_MAX_LEN = 50

SYSTEM_PROMPT = """Olet suomenkielisen teknologiapodcastin toimituksellinen avustaja.
Saat syötteenä yhden jakson litteraatin, jaettuna segmentteihin (yksi segmentti
= yksi puheenaihe). Tehtäväsi on tuottaa jakson pohjalta kolme asiaa:

1. `title` — jakson otsikko. Luettele kaikki jaksossa käsitellyt aiheet
   mahdollisimman tiiviisti. Käytä kaksoispistettä tai pilkkuja erottimena.
   Ei sisäisiä lainausmerkkejä.
2. `description` — 2–4 virkkeen kuvaus jakson sisällöstä. Ei sisäisiä
   lainausmerkkejä.
3. `body` — blogipostauksen runko Markdownina. Käsittele jokainen segmentti
   omana lukunaan. Jokaisella luvulla on oma `##`-tason väliotsikko. Kirjoita
   sujuvaa proosaa, älä luettele ranskalaisia viivoja. Älä toista otsikkoa
   tai kuvausta.

Tärkeitä kirjoitussääntöjä:

- Käytä suomen **passiivimuotoa**. Esimerkki aloituksesta: "Tässä jaksossa
  puhutaan siitä, että...". Älä kirjoita aktiivissa yksikön ensimmäisessä
  tai kolmannessa persoonassa ("minä puhun", "puhuja kertoo", "Jaakko puhuu",
  "juontaja kertoo" jne.).
- **Älä koskaan mainitse puhujaa nimeltä** (esim. "Jaakko") tai viittaa
  häneen roolinimellä ("juontaja", "isäntä", "puhuja"). Käsittele jakson
  sisältö ikään kuin se kerrottaisiin yleisesti — tekstissä ei ole
  nimettyä kertojaa.
- Äänensävyohjeessa sallittu suora lukijan puhuttelu ("sinä") on OK, mutta
  kirjoittaja itse pysyy nimettömänä ja ensimmäisen persoonan yksikkö on
  kielletty.

Noudata käyttäjän antamaa äänensävyohjetta (tone of voice) kaikissa kolmessa
tekstissä näiden sääntöjen puitteissa. Kirjoita aina suomeksi.

Palauta vastauksesi tarkalleen tässä JSON-muodossa ilman mitään selittäviä
rivejä, koodilohkoja tai tekstiä ennen tai jälkeen:

{"title": "...", "description": "...", "body": "..."}
"""


def _segment_block(stem: str, chunks: list[TranscriptChunk]) -> str:
    body = " ".join(c.text for c in chunks).strip()
    return f"## Segmentti: {stem}\n\n{body}\n"


def _build_source(
    transcripts: dict[str, list[TranscriptChunk]], order: list[str]
) -> str:
    return "\n".join(_segment_block(s, transcripts[s]) for s in order)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_response(text: str) -> dict:
    cleaned = _strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _call_claude(tone: str, source: str, model: str, max_tokens: int) -> dict:
    from anthropic import Anthropic

    client = Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": "# Äänensävy (tone of voice)\n\n" + tone.strip(),
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "# Jakson segmentit (puheen litteraatti)\n\n" + source
                ),
            }
        ],
    )
    # Anthropic SDK returns a list of content blocks; concatenate text blocks.
    parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
    return _parse_json_response("".join(parts))


def _yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _slugify(text: str, max_len: int = SLUG_MAX_LEN) -> str:
    """Lower-case ASCII slug for use in blog filenames. Transliterates
    Finnish diacritics (ä→a, ö→o, å→a) via NFKD decomposition, replaces any
    remaining non-alphanumerics with '-', and hard-truncates to `max_len`
    characters (trailing hyphen stripped). Matches the
    `YYYY-MM-DD-<slug>.md` convention used by the site."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9]+", "-", stripped).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "post"


def _render_blog_post(
    title: str, description: str, body: str, post_date: str
) -> str:
    frontmatter = (
        "---\n"
        'layout: "layouts/blog-post.njk"\n'
        f"title: {_yaml_quote(title)}\n"
        f"description: {_yaml_quote(description)}\n"
        f'date: "{post_date}"\n'
        "---\n\n"
    )
    return frontmatter + body.strip() + "\n"


def _render_shownotes(title: str, description: str) -> str:
    return f"# {title}\n\n{description.strip()}\n"


def generate_blog(
    session: Session,
    model_size: str = "small",
    llm_model: str = DEFAULT_LLM_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[Path, Path]:
    """Generate the text artifacts for an episode:

    - `<session>/output/shownotes.md` — plain-text title + description.
    - `<session>/output/<YYYY-MM-DD>-<slug>.md` — full blog post (frontmatter
      + chaptered body) matching the site template; the slug is derived from
      the generated title.

    Returns `(shownotes_path, blog_post_path)`. Transcribes on demand if the
    transcript cache is stale."""
    tone_path = require_tone_of_voice()
    tone = tone_path.read_text()

    _console.rule(f"[bold]podtool blog[/bold] {session.root.name}")
    transcripts = transcribe_session(session, model_size=model_size)
    order = [s.stem for s in session.segments]
    source = _build_source(transcripts, order)

    _console.log(f"calling {llm_model} ({len(session.segments)} segment(s))")
    result = _call_claude(tone, source, llm_model, max_tokens)

    missing = [k for k in ("title", "description", "body") if k not in result]
    if missing:
        raise ValueError(
            f"LLM response missing required keys: {', '.join(missing)}"
        )
    title = str(result["title"]).strip()
    description = str(result["description"]).strip()
    body = str(result["body"]).strip()

    post_date = date.today().isoformat()
    rendered = _render_blog_post(title, description, body, post_date)
    shownotes = _render_shownotes(title, description)

    output_dir = session.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shownotes_path = output_dir / "shownotes.md"
    shownotes_path.write_text(shownotes)

    blog_filename = f"{post_date}-{_slugify(title)}.md"
    blog_path = output_dir / blog_filename
    blog_path.write_text(rendered)

    _console.log(f"[green]wrote[/green] {shownotes_path}")
    _console.log(f"[green]wrote[/green] {blog_path}")
    _console.log(f"title       : {title}")
    snippet = description if len(description) <= 100 else description[:97] + "..."
    _console.log(f"description : {snippet}")
    return shownotes_path, blog_path
