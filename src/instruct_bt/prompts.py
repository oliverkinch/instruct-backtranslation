"""Prompt templates for the instruct-backtranslation pipeline.

Prompts are written in English (the synthesis LLM performs best with English
meta-instructions) but each prompt that produces user-facing content explicitly
requires Danish output.

Templates are organised by **source type** so that the framing matches the
kind of text being processed (encyclopedic, speech, literary, etc.).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source-type registry: maps DynaWord subset names → source types
# ---------------------------------------------------------------------------

#: Recognised source types.  Every subset that is not listed here falls back
#: to ``"knowledge"`` (the Wikipedia-style default).
#: The six DynaWord subsets we process from HuggingFace.
DYNAWORD_SUBSETS: tuple[str, ...] = (
    "wikipedia",
    "danske-taler",
    "ft",
    "nordjyllandnews",
    "miljoeportalen",
    "ai-aktindsigt",
)

SOURCE_TYPES: dict[str, str] = {
    # knowledge / encyclopedic
    "wikipedia": "knowledge",
    # speeches & parliamentary records
    "danske-taler": "speech",
    "ft": "speech",
    # news
    "nordjyllandnews": "news",
    "dkmedier": "news",
    # governmental / policy
    "miljoeportalen": "government",
    "ai-aktindsigt": "government",
}

DEFAULT_SOURCE_TYPE = "knowledge"


def source_type_for_subset(subset: str) -> str:
    """Return the source type for a given DynaWord subset name."""
    return SOURCE_TYPES.get(subset, DEFAULT_SOURCE_TYPE)


# ===================================================================
# SELECT TEMPLATES — one per source type
# ===================================================================
#
# Each template receives ``{paragraph}`` and must instruct the LLM to
# either extract the best verbatim passage or output ``SKIP``.
# ===================================================================

_SELECT_KNOWLEDGE = """\
[Text]
{paragraph}


[Task]
The text above is a section from a Danish article. Your job is to extract \
the best passage from this text that reads naturally as a response from a \
helpful Danish AI assistant.

A good passage EXPLAINS, CONTEXTUALIZES, or TEACHES something — it helps \
a reader understand WHY or HOW, not just WHAT. Imagine a curious person \
asked a question and a knowledgeable friend answered. The passage should \
feel like that answer.

Rules:
    - Copy the passage EXACTLY from the text — do not rephrase, summarize, \
or add anything. Every word must appear in the original.
    - The passage may be one or several consecutive paragraphs (1-4 \
paragraphs, roughly 50-500 words). Include enough context that the \
passage forms a complete, satisfying answer — not a fragment.
    - The passage must be self-contained: a reader should understand it \
without needing the rest of the article.

SKIP the text (output exactly: SKIP) if ANY of these apply:
    - Only states bare facts without explanation (e.g. "X er en by i Y \
med Z indbyggere" with no further context).
    - Is a list of names, dates, track listings, cast members, or awards.
    - Is a chronological dump of events without analysis or explanation \
(e.g. "I 2004 fik partiet X%... I 2008... I 2012...").
    - Reads like a dictionary entry or a database record.
    - Is a biographical CV: a sequence of positions, dates, and career \
moves without insight into what the person actually did or why it \
matters (e.g. "1968-71 Politiken, 1972 Grønlands Radio, 1973-76 DR...").
    - Is a geographic or infrastructure inventory: describes location, \
boundaries, neighbouring areas, or physical features without explaining \
anything about the place's significance.
    - Is sports or election results presented season-by-season or \
year-by-year without analysis.
    - Is too short to be a satisfying answer — a single sentence or two \
brief sentences that merely identify a subject (e.g. "X var en dansk \
forfatter" or "X er en film fra 1992").
    - Is a technical specification list (dimensions, ranges, capacities) \
without explaining what the numbers mean or why they matter.
    - Is primarily a direct quotation (wrapped in quotation marks) rather \
than explanatory prose. An AI assistant explains in its own words — it \
does not just paste a quote.
    - Is a band, album, or film description that only lists members, \
songs, or credits without explaining significance.

When in doubt, SKIP. It is better to be strict — we want quality over \
quantity.

Output ONLY the extracted passage (or SKIP). No commentary."""


_SELECT_SPEECH = """\
[Text]
{paragraph}


[Task]
The text above is an excerpt from a Danish speech or parliamentary debate. \
Your job is to extract the best passage that conveys a clear argument, \
opinion, or message — something that would make sense as the output if a \
user asked an AI to write or draft a speech on a specific topic.

A good passage has rhetorical force: it argues a point, appeals to values, \
proposes action, or paints a vision. It should read as a coherent, \
self-contained piece of oratory — not a procedural remark or roll-call.

Rules:
    - Copy the passage EXACTLY from the text — do not rephrase, summarize, \
or add anything. Every word must appear in the original.
    - The passage may span 1-4 consecutive paragraphs (roughly 50-500 words).
    - The passage must be self-contained: a reader should understand the \
argument without needing the rest of the speech.

SKIP the text (output exactly: SKIP) if ANY of these apply:
    - Procedural or administrative remarks (opening/closing a session, \
announcing agenda items, vote tallies).
    - Back-and-forth debate fragments that only make sense in dialogue context.
    - Lists of names, dates, numbers, or legislative references without argument.
    - Too short or fragmented to convey a coherent point (fewer than \
4-5 sentences).
    - Thanking the chair, greeting the audience, or other formulaic \
pleasantries without substantive content.
    - References to specific speakers, interruptions, or procedural \
exchanges that would not make sense outside the debate chamber.

When in doubt, SKIP. Quality over quantity.

Output ONLY the extracted passage (or SKIP). No commentary."""


_SELECT_NEWS = """\
[Text]
{paragraph}


[Task]
The text above is an excerpt from a Danish news article. Your job is to \
extract the best passage that reports, explains, or analyses an event — \
something that would make sense as the output if a user asked an AI to \
write a news article about a specific topic.

A good passage tells the reader WHAT happened, WHY it matters, and \
includes context or quotes that make it informative and engaging.

Rules:
    - Copy the passage EXACTLY from the text — do not rephrase, summarize, \
or add anything. Every word must appear in the original.
    - The passage may span 1-4 consecutive paragraphs (roughly 50-500 words).
    - The passage must be self-contained: a reader should understand the \
story without needing the rest of the article.

SKIP the text (output exactly: SKIP) if ANY of these apply:
    - Only states bare facts without context (scores, weather, stock prices).
    - Is a list of events without reporting.
    - Is too short to convey a meaningful story (fewer than 4-5 sentences).
    - Contains mostly metadata (dates, bylines, tags) rather than content.
    - Is primarily direct quotes strung together without journalistic \
framing or context — the passage should read as a coherent article \
excerpt, not a transcript of quotes.
    - Describes a single minor event without broader context or \
significance (e.g. a brief accident report with no analysis).

When in doubt, SKIP. Quality over quantity.

Output ONLY the extracted passage (or SKIP). No commentary."""


_SELECT_GOVERNMENT = """\
[Text]
{paragraph}


[Task]
The text above is an excerpt from a Danish governmental or public-sector \
document (policy report, regulation, municipal website, etc.). Your job \
is to extract the best passage that explains a policy, regulation, \
service, or public issue in a way that is informative and useful — \
something that would make sense as a response from a knowledgeable \
AI assistant about Danish public administration.

A good passage explains HOW something works, WHY a regulation exists, \
or WHAT citizens can expect — not just stating a rule number or \
listing administrative units.

Rules:
    - Copy the passage EXACTLY from the text — do not rephrase, summarize, \
or add anything. Every word must appear in the original.
    - The passage may span 1-4 consecutive paragraphs (roughly 50-500 words).
    - The passage must be self-contained.

SKIP the text (output exactly: SKIP) if ANY of these apply:
    - Pure tables, numbers, or data without explanation.
    - Lists of organizational units, addresses, or contact information.
    - Boilerplate legal disclaimers or cookie notices.
    - Too technical or fragmented to be useful to a general reader.
    - Vague procedural descriptions that only reference other documents \
or sections (e.g. "This is described in section X" or "will be \
addressed in VVM assessments") without giving concrete, useful \
information a citizen could act on.
    - Generic process descriptions that could apply to any topic and \
do not convey specific knowledge (e.g. "measures will be planned \
and implemented as needed").
    - Too short to be a satisfying answer — fewer than 4-5 sentences.

When in doubt, SKIP. Quality over quantity.

Output ONLY the extracted passage (or SKIP). No commentary."""


SELECT_TEMPLATES: dict[str, str] = {
    "knowledge": _SELECT_KNOWLEDGE,
    "speech": _SELECT_SPEECH,
    "news": _SELECT_NEWS,
    "government": _SELECT_GOVERNMENT,
}


# ===================================================================
# INSTRUCTION TEMPLATES — one per source type
# ===================================================================
#
# Each template receives ``{paragraph}`` and must instruct the LLM to
# generate a short, natural Danish user message that would produce the
# given text as output.
# ===================================================================

_INSTRUCTION_KNOWLEDGE_BASE = """\
[Response from a Danish AI assistant]
{paragraph}


[Task]
A user sent a message to a Danish AI assistant and received the response \
above. Write the message the user most likely sent.

Rules:
    - Write in Danish.
    - Do NOT mention details that only appear in the response. The user \
asks BEFORE seeing the answer.
    - Do NOT add preamble or explanation — output only the user's message.
"""

_INSTRUCTION_KNOWLEDGE_A = _INSTRUCTION_KNOWLEDGE_BASE + """\
    - Write a SHORT direct question — between 5 and 10 words. Just the \
core question, nothing else. No framing, no context, no "Kan du fortælle \
mig om...". Examples:
    - "Hvad er fotosyntese?"
    - "Hvorfor opstod reformationen?"
    - "Hvem var wagrierne?"
"""

_INSTRUCTION_KNOWLEDGE_B = _INSTRUCTION_KNOWLEDGE_BASE + """\
    - Write ONE single sentence — a request that names the topic and \
specifies what kind of information the user wants. It should be longer \
than a bare question but NOT include personal backstory or context. \
Aim for 12-20 words.
    - Do NOT write a bare 3-6 word question. Do NOT start with "Jeg" \
or add personal context.
    - Examples:
    - "Kan du give mig en kort introduktion til kvantemekanik og dens grundprincipper?"
    - "Hvad er egentlig forskellen mellem en galakse og en stjernehob i astronomien?"
    - "Forklar mig venligst hvad den westfalske fred betød for Europa efter Trediveårskrigen"
"""

_INSTRUCTION_KNOWLEDGE_C = _INSTRUCTION_KNOWLEDGE_BASE + """\
    - Write a LONGER conversational message (15-30 words) where the user \
first gives some personal context and THEN asks the question. It should \
feel like a real person explaining their situation before asking. Examples:
    - "Jeg sidder og skriver en opgave om slaviske folk i Nordeuropa — kan du fortælle mig om wagrierne?"
    - "Jeg læste noget om vindenergi i dag men forstår det ikke helt, kan du forklare?"
    - "Min søn spurgte mig om rumfart og jeg kunne ikke rigtig svare — hvad er en rumstation?"
"""


_INSTRUCTION_SPEECH_BASE = """\
[Output from a Danish AI assistant]
{paragraph}


[Task]
A user asked a Danish AI assistant to write or draft something, and the \
assistant produced the text above. The text is a passage from a speech \
or parliamentary address. Write the user message that most likely \
prompted the assistant to generate this text.

Rules:
    - Write in Danish.
    - The message should be a writing or drafting request.
    - Capture the TOPIC or THEME of the passage, not specific details \
that only appear in the output.
    - Do NOT add preamble or explanation — output only the user's message.
"""

_INSTRUCTION_SPEECH_A = _INSTRUCTION_SPEECH_BASE + """\
    - Write a SHORT command — between 5 and 12 words. Just the request, \
nothing else. Vary your phrasing — do NOT always use "Skriv". Examples:
    - "Skriv en tale om grøn transport"
    - "Formuler et kort indlæg om ældreplejen"
    - "Lav et udkast til en tale om frivilligt arbejde"
"""

_INSTRUCTION_SPEECH_B = _INSTRUCTION_SPEECH_BASE + """\
    - Write ONE single sentence — a drafting request that names the topic \
and what angle or argument to take. It should be longer than a bare \
command but NOT include personal backstory. Aim for 12-20 words.
    - Do NOT write a bare 3-7 word command. Do NOT start with "Jeg" \
or add personal context about your situation.
    - Examples:
    - "Kan du skrive et indlæg der argumenterer for flere investeringer i den danske folkeskole?"
    - "Hjælp mig med at formulere et overbevisende argument for mere vedvarende energi i Danmark"
    - "Skriv et debatindlæg der forsvarer retten til privatliv i en digital tidsalder"
"""

_INSTRUCTION_SPEECH_C = _INSTRUCTION_SPEECH_BASE + """\
    - Write a LONGER conversational message (20-35 words) where the user \
first gives personal context about their situation and THEN makes the \
request. It should feel like a real person explaining why they need help. \
Examples:
    - "Jeg skal holde en tale til et arrangement om ligestilling næste uge, kan du hjælpe mig med et udkast?"
    - "Jeg sidder og forbereder et debatindlæg om sundhedsreformen og mangler inspiration — kan du give mig et bud?"
    - "Vi har en debat om klimapolitik i klassen i morgen og jeg skal argumentere for grøn omstilling, kan du hjælpe?"
"""


_INSTRUCTION_NEWS_BASE = """\
[Output from a Danish AI assistant]
{paragraph}


[Task]
A user asked a Danish AI assistant to write something, and the assistant \
produced the text above. The text is a passage from a Danish news \
article. Write the user message that most likely prompted the assistant \
to generate this text.

Rules:
    - Write in Danish.
    - The message should be a request to write, summarise, or report on \
a news topic.
    - Capture the TOPIC of the article, not specific details from the output.
    - Do NOT add preamble or explanation — output only the user's message.
"""

_INSTRUCTION_NEWS_A = _INSTRUCTION_NEWS_BASE + """\
    - Write a SHORT command — between 5 and 12 words. Just the core \
request. Vary your phrasing — do NOT always use "Skriv" or "Kan du lave". \
Examples:
    - "Skriv en kort nyhedsartikel om kommunalvalget i Nordjylland"
    - "Giv mig et nyhedsreferat om oversvømmelserne i Jylland"
    - "Fortæl om hvad der skete ved klimatopmødet"
"""

_INSTRUCTION_NEWS_B = _INSTRUCTION_NEWS_BASE + """\
    - Write ONE single sentence — a writing request that names the news \
topic and what kind of article you want. It should be longer than a bare \
command but NOT include personal backstory. Aim for 12-20 words.
    - Do NOT write a bare 3-7 word command. Do NOT start with "Jeg" \
or add personal context about your situation.
    - Examples:
    - "Kan du skrive en kort nyhedsartikel om de seneste resultater i Superligaen?"
    - "Hjælp mig med at skrive en artikel om den nye broforbindelse over Femern Bælt"
    - "Lav en kort reportage om det lokale fodboldhold der vandt pokalen i weekenden"
"""

_INSTRUCTION_NEWS_C = _INSTRUCTION_NEWS_BASE + """\
    - Write a LONGER conversational message (20-35 words) where the user \
first gives personal context about their situation and THEN makes the \
request. It should feel like a real person explaining why they need this. \
Examples:
    - "Jeg skal skrive en artikel om lukningerne i detailhandlen til min blog, kan du give mig et udkast?"
    - "Jeg sidder og mangler en vinkel på en artikel om boligpriserne i København — kan du hjælpe?"
    - "Min redaktør har bedt mig om en artikel om det nye hospitalsprojekt, kan du skrive et udkast?"
"""


_INSTRUCTION_GOVERNMENT_BASE = """\
[Response from a Danish AI assistant]
{paragraph}


[Task]
A user sent a message to a Danish AI assistant and received the response \
above. The text comes from a Danish governmental or public-sector source. \
Write the message the user most likely sent.

Rules:
    - Write in Danish.
    - The message can be a question about rules, policies, or public \
services — or a request to explain or summarise something.
    - Do NOT mention details that only appear in the response.
    - Do NOT add preamble or explanation — output only the user's message.
"""

_INSTRUCTION_GOVERNMENT_A = _INSTRUCTION_GOVERNMENT_BASE + """\
    - Write a SHORT direct question — between 5 and 10 words. Just the \
core question, nothing else. Examples:
    - "Hvad er reglerne for kvælstofudledning?"
    - "Hvornår skal man indberette moms?"
    - "Hvad er Natura 2000?"
"""

_INSTRUCTION_GOVERNMENT_B = _INSTRUCTION_GOVERNMENT_BASE + """\
    - Write ONE single sentence — a request that names the topic and \
specifies what kind of information you need. It should be longer than a \
bare question but NOT include personal backstory. Aim for 10-20 words.
    - Do NOT write a bare 3-6 word question. Do NOT start with "Jeg" \
or add personal context about your situation.
    - Examples:
    - "Kan du fortælle mig om kommunens tilbud og støttemuligheder for handicappede borgere?"
    - "Giv mig et overblik over reglerne for sygedagpenge og hvornår man er berettiget"
    - "Forklar venligst hvad den nye klimahandlingsplan indebærer for danske kommuner"
"""

_INSTRUCTION_GOVERNMENT_C = _INSTRUCTION_GOVERNMENT_BASE + """\
    - Write a LONGER conversational message (15-30 words) where the user \
first gives personal context about their situation and THEN asks the \
question. It should feel like a real person explaining why they need \
this information. Examples:
    - "Jeg skal søge om byggetilladelse til en tilbygning og er usikker på processen — hvordan gør man?"
    - "Jeg er i tvivl om mine rettigheder som lejer, min udlejer vil hæve huslejen — kan du hjælpe?"
    - "Jeg har lige startet en virksomhed og er usikker på reglerne for ansættelse — hvad skal jeg vide?"
"""


INSTRUCTION_TEMPLATES: dict[str, dict[str, str]] = {
    "knowledge": {
        "A": _INSTRUCTION_KNOWLEDGE_A,
        "B": _INSTRUCTION_KNOWLEDGE_B,
        "C": _INSTRUCTION_KNOWLEDGE_C,
    },
    "speech": {
        "A": _INSTRUCTION_SPEECH_A,
        "B": _INSTRUCTION_SPEECH_B,
        "C": _INSTRUCTION_SPEECH_C,
    },
    "news": {
        "A": _INSTRUCTION_NEWS_A,
        "B": _INSTRUCTION_NEWS_B,
        "C": _INSTRUCTION_NEWS_C,
    },
    "government": {
        "A": _INSTRUCTION_GOVERNMENT_A,
        "B": _INSTRUCTION_GOVERNMENT_B,
        "C": _INSTRUCTION_GOVERNMENT_C,
    },
}

#: Format keys for random selection in generate.py
INSTRUCTION_FORMATS = ("A", "B", "C")

# ---------------------------------------------------------------------------
# System messages
# ---------------------------------------------------------------------------

#: System message for the SELECT stage (passage extraction / quality filtering).
SELECT_SYSTEM_MSG = (
    "You are an expert text evaluator. You assess Danish texts and extract "
    "passages that would work well as AI assistant responses. You are strict "
    "about quality — you prefer to skip a text rather than select a mediocre "
    "passage. You always copy text verbatim and never rephrase."
)

#: System message for the GENERATE stage (instruction backtranslation).
GENERATE_SYSTEM_MSG = (
    "You are a Danish language expert specialising in writing realistic user "
    "messages. You write natural, varied Danish — the kind of messages real "
    "people send to AI assistants. You never explain your reasoning and never "
    "output anything except the user message itself."
)


