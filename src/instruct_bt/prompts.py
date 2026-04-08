"""Prompt templates for the instruct-backtranslation pipeline.

Prompts are written in English (the synthesis LLM performs best with English
meta-instructions) but each prompt that produces user-facing content explicitly
requires Danish output.
"""

# ---------------------------------------------------------------------------
# Paragraph selection: given a heuristically-extracted section, have the LLM
# pick the best passage that would work as a chatbot response.
# ---------------------------------------------------------------------------

SELECT_TEMPLATE = """\
[Text]
{paragraph}


[Task]
The text above is a section from a Danish article. Your job is to extract \
the single best passage from this text that could serve as a natural response \
from a Danish AI assistant.

Rules:
    - Copy the passage EXACTLY from the text — do not rephrase, summarize, \
or add anything.
    - The passage should be self-contained: a reader should understand it \
without needing the rest of the article.
    - The passage should be informative and read naturally as an answer to \
a question someone might ask a chatbot.
    - Prefer concise passages (1-3 sentences, roughly 50-200 words). \
Skip overly long or dense passages.
    - Skip passages that are just lists of names, dates, track listings, \
or cast members.
    - If no part of the text works well as a chatbot response, output \
exactly: SKIP
    - Output ONLY the extracted passage (or SKIP). No commentary."""

# ---------------------------------------------------------------------------
# Instruction generation: given a selected paragraph, generate a natural
# user question/request that would elicit it as a chatbot response.
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = """\
[Response from a Danish AI assistant]
{paragraph}


[Task]
A user sent a message to a Danish AI assistant and received the response \
above. Write the message the user most likely sent.

Rules:
    - Write in Danish.
    - Be brief and natural — the way a real person types a question into \
a chatbot (typically 3-15 words).
    - Do NOT mention details that only appear in the response. The user \
asks BEFORE seeing the answer.
    - Do NOT add preamble or explanation — output only the user's message.

Examples of natural user messages:
    - "Hvad er fotosyntese?"
    - "Fortæl mig om Den Mørke Middelalder"
    - "Hvor ligger Wagga Wagga?"
    - "Hvem var Edgar Wallace?"
    - "Hvad er reglerne for stemmeret i Kuwait?"
    - "Forklar kort hvad nyateisme er"
"""

# ---------------------------------------------------------------------------
# System messages
# ---------------------------------------------------------------------------

SYSTEM_MSG = "You are a helpful assistant."
