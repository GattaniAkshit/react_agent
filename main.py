"""Simple ReAct-style agent demo.

Overview:
- Demonstrates a minimal ReAct loop: the LLM thinks, acts (via tools), observes, and answers.
- Tools: `calculator(expr)` and `search(query)`.
- Controller: prompts for one iteration per turn; injects Observations; enforces guardrails.

This file also logs a detailed report to a rotating file while keeping console output
minimal and intuitive for non-technical viewers.
"""

import os
import ast
import time
import logging
import argparse
from groq import Groq
import operator as op
from ddgs import DDGS
from dotenv import load_dotenv
from typing import Dict, List, Optional
from logging.handlers import RotatingFileHandler

# Loading env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
DEFAULT_SEARCH_RESULT = os.getenv("DEFAULT_SEARCH_RESULT")
MODEL_ID = os.getenv("MODEL_ID")

# Allowed operators
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg
}

EXPECTED_KEYS = ["Thought", "Action", "Action input", "Observation", "Final Answer"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

#######################
### Helper Functions###
#######################

def _setup_logger() -> logging.Logger:
    """Configure a rotating file logger for detailed demo reports.

    Returns:
        A configured logger instance writing to `report_log.txt` with rotation.
    """
    logger = logging.getLogger("agent_demo")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            "report_log.txt", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = _setup_logger()


def _divider(title: str = "") -> None:
    """Print a minimal section divider to console for readability."""
    line = "=" * 40
    if title:
        print(f"\n{line} {title} {line}")
    else:
        print(f"\n{line*2}")


def _truncate(text: str, max_len: int = 400) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."

def _print_turn(turn_idx: int, raw_response: str, parsed: Dict[str, str]) -> None:
    """Log full details to the report and print a minimal summary to console.

    Report markers (file):
    - [REPORT|TURN]: turn header
    - [REPORT|RAW]: raw LLM response
    - [REPORT|PARSED]: parsed keys and values
    """
    # Detailed file log
    LOGGER.info("[REPORT|TURN] #%s", turn_idx)
    LOGGER.info("[REPORT|RAW] %s", raw_response)
    for k in EXPECTED_KEYS:
        LOGGER.info("[REPORT|PARSED] %s=%s", k, parsed.get(k, ""))

    # Minimal console summary
    _divider(f"Turn {turn_idx}")
    thought = _truncate(parsed.get("Thought", ""), 140)
    action = parsed.get("Action", "")
    final_present = "Yes" if parsed.get("Final Answer", "").strip() else "No"
    print(f"Thought: {thought}")
    print(f"Action: {action}")
    print(f"Has Final Answer: {final_present}")

def _print_tool(tool_name: str, tool_input: str, tool_result: str) -> None:
    """Log tool call details and print a minimal console line."""
    LOGGER.info("[REPORT|TOOL] name=%s input=%s", tool_name, tool_input)
    LOGGER.info("[REPORT|TOOL|RESULT] %s", tool_result)
    print(f"Tool: {tool_name} | Done")

def _print_transcript(messages: List[Dict]) -> None:
    """Write a compact transcript to the report log with unique markers."""
    LOGGER.info("[REPORT|TRANSCRIPT|BEGIN]")
    idx = 1
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue  # do not expose system prompt in report
        content = m.get("content", "")
        LOGGER.info("[REPORT|TRANSCRIPT|%02d] role=%s content=%s", idx, role.upper(), content)
        idx += 1
    LOGGER.info("[REPORT|TRANSCRIPT|END]")

def search(query: str, max_results: Optional[int] = 3) -> str:
    """Call DuckDuckGo API and aggregate top results into a concise Observation.

    Args:
        query: search query (str)
        max_results: optional to limit results retrieved (default: 3)

    Returns:
        str: aggregated search summary including titles, snippets, and urls (when available)
    """
    LOGGER.info("[REPORT|TOOL|ENTER] search query=%s", query)
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query=query, max_results=max_results)]
    except Exception:
        results = []

    if not results:
        return DEFAULT_SEARCH_RESULT or "No results found"

    # Build a compact, structured observation from the top results
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        title = str(r.get('title', '')).strip()
        body = str(r.get('body', '')).strip()
        href = str(r.get('href', '')).strip()
        # Prefer including title when present
        parts = []
        if title:
            parts.append(f"{i}. {title}")
        else:
            parts.append(f"{i}.")
        if body:
            parts.append(body)
        if href:
            parts.append(f"URL: {href}")
        lines.append(" — ".join(parts))

    observation = "\n".join(lines)
    # Cap length to keep the conversation efficient; the full text is often not needed
    if len(observation) > 1800:
        observation = observation[:1797] + "..."
    LOGGER.info("[REPORT|TOOL|EXIT] search results_compiled=%s", observation)
    return observation


def calculator(expr: str) -> str:
    """Safely evaluate a basic arithmetic expression and return the result as string.

    Supports +, -, *, /, ** and unary - using a safe AST evaluation with a
    restricted operator set defined in `ALLOWED_OPERATORS`.
    """
    LOGGER.info("[REPORT|TOOL|ENTER] calculator expr=%s", expr)
    def _eval(node):
        if isinstance(node, ast.Constant): # numbers
            return node.value
        elif isinstance(node, ast.BinOp): # binary operations (e.g., 2+3)
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Unsupported operator")
            return ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp): # unary operations (e.g. -3)
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Unsupported unary operator")
            return ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
        else:
            raise ValueError("Unsupported expression")
    
    node = ast.parse(expr, mode="eval").body
    result = str(_eval(node))
    LOGGER.info("[REPORT|TOOL|EXIT] calculator result=%s", result)
    return result


def chat(messages: List[Dict]) -> str:
    """Helper to interact with LLM with basic 429 retry/backoff.

    Args:
        messages: list of dictionaries holding the interaction with the llm

    Returns:
        response: react agent response
    """
    # Basic retry with backoff to handle transient rate limits (429)
    last_err = None
    for attempt in range(3):
        try:
            response_object = client.chat.completions.create(model=MODEL_ID, messages=messages)
            response = response_object.choices[0].message.content if response_object.choices[0].message.content is not None else ""
            LOGGER.info("[REPORT|LLM|OK] chars=%d", len(response))
            return response
        except Exception as e:
            last_err = e
            if "429" in str(e):
                time.sleep(2 ** attempt)
                continue
            raise
    if last_err is not None:
        LOGGER.error("[REPORT|LLM|ERROR] %s", str(last_err))
        raise last_err

# Tool name to tool call map
TOOL_NAME_MAP = {
    "calculator": calculator,
    "search": search
}

def parse_response(resp: str) -> dict:
    """Parse the LLM response lines into the expected key/value block.

    Be forgiving with common key variants; prefer keeping the first non-empty
    value for each key to avoid clobbering by later repeats.
    """
    parsed = {k: '' for k in EXPECTED_KEYS}
    key_map = {
        'thought': 'Thought',
        'action': 'Action',
        'action input': 'Action input',
        'action_input': 'Action input',
        'final answer': 'Final Answer',
        'final_answer': 'Final Answer',
        'observation': 'Observation',
    }
    for raw_line in resp.split("\n"):
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_norm = key.strip().lower()
        value = value.strip()
        # Unwrap fully quoted empty values like '' or "" or `` and normalize
        def _unwrap(v: str) -> str:
            v = v.strip()
            if (len(v) >= 2 and ((v[0] == v[-1]) and v[0] in ("'", '"', '`'))):
                return v[1:-1].strip()
            return v
        value_unwrapped = _unwrap(value)
        # We keep the first non-empty value for a key; ignore later empty overwrites
        if key_norm in key_map:
            target = key_map[key_norm]
            if parsed[target] == '' and value_unwrapped != '':
                parsed[target] = value_unwrapped
        elif key.strip() in parsed:
            target = key.strip()
            if parsed[target] == '' and value_unwrapped != '':
                parsed[target] = value_unwrapped
    return parsed

#######################
### Main Function###
#######################

def main() -> None:
    """Entry point for the demo app flow.

    Flow:
    1) Seed messages with system prompt and user query.
    2) Loop until Final Answer or `max_turns`.
    3) On Action, run the tool and append an Observation.
    4) Apply guardrails and provide corrective feedback when needed.

    Console output: minimal, narrative flow.
    Log file (report_log.txt): detailed, structured with unique markers.
    """
    parser = argparse.ArgumentParser(description="ReAct agent simple demo")

    parser.add_argument("query", type=str, help="Input any mathematical or search query")

    args = parser.parse_args()

    query = args.query

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }, 
        {
            "role": "user", 
            "content": query
        }
    ]

    max_turns = 6
    turns = 0
    final_answer = ""
    search_used = False
    add_finalize_note = False  # add one-time clamp after a tool call

    while turns < max_turns and not final_answer:
        if add_finalize_note:
            messages.append({
                "role": "system",
                "content": (
                    "Observation has been provided. Do not choose another Action in this turn. "
                    "Provide a Final Answer using the Observation. Output exactly these 5 keys once, "
                    "with Action and Action input set to ''."
                )
            })
            add_finalize_note = False

        agent_response_str = chat(messages=messages)
        try:
            agent_response_dict = parse_response(agent_response_str)
        except Exception as e:
            print(agent_response_str)
            raise e
        _print_turn(turns + 1, agent_response_str, agent_response_dict)

        # Append the model's schema response to maintain conversation state
        messages.append({"role": "assistant", "content": agent_response_str})

        # Final answer?
        fa = agent_response_dict.get("Final Answer", "").strip()
        if fa:
            final_answer = fa
            break

        # Handle action if present
        action = agent_response_dict.get("Action", "").strip()
        if (action.startswith("'") and action.endswith("'")) or (action.startswith('"') and action.endswith('"')):
            action = action[1:-1].strip()
        if action:
            tool_name = action
            if tool_name.startswith("[") and tool_name.endswith("]"):
                tool_name = tool_name[1:-1]
            tool_name = tool_name.strip().lower()
            tool_input = agent_response_dict.get("Action input", "").strip()
            if (tool_input.startswith("`") and tool_input.endswith("`")) or \
               (tool_input.startswith("'") and tool_input.endswith("'")) or \
               (tool_input.startswith('"') and tool_input.endswith('"')):
                tool_input = tool_input[1:-1]

            if tool_name in TOOL_NAME_MAP:
                # Allow calculator multiple times; restrict search to once
                if tool_name == "search" and search_used:
                    # nudge to finalize without dispatching and clamp next turn
                    messages.append({
                        "role": "system",
                        "content": (
                            "An Observation from search has already been provided. Do not call search again. "
                            "Use the Observation to produce a Final Answer."
                        )
                    })
                    add_finalize_note = True
                else:
                    tool_result = TOOL_NAME_MAP[tool_name](tool_input)
                    _print_tool(tool_name, tool_input, str(tool_result))
                    messages.append({"role": "assistant", "content": "Observation: " + str(tool_result)})
                    if tool_name == "search":
                        search_used = True
                    add_finalize_note = True

        turns += 1

    # Write full transcript to log; show a simple final line to console
    _print_transcript(messages)
    LOGGER.info("[REPORT|FINAL] %s", final_answer)
    _divider("Final Answer")
    print(final_answer)


if __name__ == "__main__":
    main()
