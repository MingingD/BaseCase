import os
import getpass
from infosci_spark_client import LLMClient
from dotenv import load_dotenv

import json

# Local testing — remove when you need.
with open("/home/daming108/BaseCase/src/cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

load_dotenv()

key = os.environ.get("SPARK_API_KEY")
if not key:
    key = getpass.getpass("SPARK_API_KEY: ")

client = LLMClient(api_key=key)


def run_case_rag(user_query: str, case_context: str, case_name: str) -> str:
    q = (user_query or "").strip()
    ctx = (case_context or "").strip()
    name = (case_name or "").strip() or "Unknown case"

    if not q:
        return "No question was provided."
    if not ctx:
        return "No case text was provided; nothing to answer from."

    rag_system = f"""
You are a legal search assistant. You must ground every factual claim about this case in the opinion text below. Do not invent facts, citations, or holdings.

Case caption: {name}

Rules:
- The case was retrieved by similarity search, so assume there is at least some connection to the user's query.
- Answer from the provided opinion text first. If the user's question is only weakly related to this case, try your best to connect the user's question to this case.
- If the text does not directly answer part of the question, add a clearly labeled inference that connects the case logic/facts to the user's scenario.
- Include at least one short quoted phrase (in quotation marks) from the opinion when you state what the court did or held, unless the excerpt is too short; then paraphrase closely and say you are paraphrasing.
- Do not give strategic or personal legal advice. This is general information from the opinion, not advice for the user's situation.
- End with a one-line disclaimer that this is not legal advice.

Use exactly this structure and these Markdown headings (in English):

## Relevance
Less than 3 sentences: does this excerpt speak to the user's question, and how directly?

## What the opinion says
First, 1-2 sentences description of the case. Second, facts and reasoning supported by the text, with at least one quote where possible.

## Connection to your question
Explain the strongest possible connection between this case and the user's scenario. If direct support is weak, use analogy and label it as an inference rather than a direct holding.

## Answer
Direct response to the user's question. Prioritize direct support from the opinion; where needed, add one short labeled inference from the case's logic.

## Limits
What this text does not establish or what remains unclear.

## Disclaimer
One sentence: informational only; not legal advice.
""".strip()

    prompt = [
        {"role": "system", "content": rag_system},
        {
            "role": "user",
            "content": f"User query:\n{q}\n\nOpinion text:\n\n{ctx}",
        },
    ]

    try:
        response = client.chat(prompt, stream=False, show_thinking=False)
        content = response.get("content")
        if content is None:
            return "The model returned no text. Try again."
        return content
    except Exception as e:
        return f"Could not get a model response ({type(e).__name__}). Try again later."


if __name__ == "__main__":
    user_query = "I copy other people's code and use it in my own project. Is that legal?"
    case_context = cases[2]["text"]
    case_name = cases[2]["case_name"]
    print(run_case_rag(user_query, case_context, case_name))
