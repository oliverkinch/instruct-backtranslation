"""Prompt templates for instruction backtranslation.

The core idea: given a real Danish paragraph, generate the instruction/question
that a user would have typed to receive this paragraph as a response from a
Danish AI assistant.

Prompts are written in English (the synthesis LLM performs best with English
meta-instructions) but explicitly require Danish output.
"""

# ---------------------------------------------------------------------------
# Instruction generation: given a paragraph, generate the matching prompt
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = """\
[Text]
{paragraph}


[Task]
The text above is a paragraph from a Danish text. Imagine you are a user \
of a Danish AI assistant. Your task is to write the most likely question \
or request that a user would send to the assistant, such that the text \
above would be an excellent response.

Follow these rules:
    - The question/request must be in Danish.
    - The question/request should be natural — something a real person \
would actually ask.
    - The question/request should be specific enough that the text above \
is a fitting and complete answer.
    - Do not reference the text directly (e.g. do not say "ifølge teksten" \
or "i ovenstående").
    - Do not include any preamble or explanation — output only the \
question/request.
    - Keep the question/request concise (at most {len_limit} words)."""

# ---------------------------------------------------------------------------
# System messages
# ---------------------------------------------------------------------------

SYSTEM_MSG = "You are a helpful assistant."
