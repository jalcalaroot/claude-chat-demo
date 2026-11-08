# Claude Chat Demo

A small local chatbot that calls the Claude API with an API key. It uses
Streamlit for the page and the official `anthropic` Python SDK.

## Features

- Replies appear as they are generated (streaming).
- The conversation keeps its context across turns.
- Editable system prompt and a choice of three models.
- The model that answered, as reported by the API, under each reply.
- Estimated cost per conversation, with a spend limit.
- Prompt caching for the repeated part of the conversation.
- Web search when a question needs current information, with links to the sources.

## Run it

Requires Python 3.12 or later.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
.venv/bin/streamlit run app.py
```

Open <http://localhost:8501>.

## Controls

| Control | Value | Where |
|---|---|---|
| API key only in the server process, from `ANTHROPIC_API_KEY` | The browser never receives it | `app.py` |
| Listens on `127.0.0.1` only | Not reachable from other machines | `.streamlit/config.toml` |
| Allowed models | Opus 5.5, Sonnet 5.5, Haiku 5.5 | `chat.py` |
| Spend limit per conversation | USD 1.00 by default | `app.py`, `chat.py` |
| Web searches per reply | 3 at most | `chat.py` |
| Images in replies are not rendered | Web content can steer a reply, and an image URL would be fetched by the browser and could leak conversation text | `chat.py` |
| Sources limited to `http` and `https` links, with titles escaped | Page titles are untrusted text | `chat.py` |
| Message and conversation size | 8,000 characters, 30 turns | `chat.py` |

The cost is an estimate from the token counts of each reply, public list prices
and USD 0.01 per web search. Search results stay in the conversation, so they
add input tokens on later turns. If a reply is declined and retried on a fallback model, the real charge
uses that model's prices.

The app has no login, so do not expose it to the internet.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

The tests use a mocked client and never call the API.

## License

[MIT](LICENSE)
