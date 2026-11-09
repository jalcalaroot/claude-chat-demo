# Verification

Results from 2026-11-08, on the code in this repository, with `anthropic` 2.3.0,
`streamlit` 1.66.1 and Python 3.14.

## Automated tests

`pytest` runs 19 tests against a mocked client and never calls the API. They
cover the model allowlist, request parameters, cost calculation, web search
limits, resuming a paused search, image neutralization, source filtering, and
the whole Streamlit script for normal, declined and truncated replies.

## Browser tests against the real API

The app ran locally and was driven in headless Chrome with Playwright, using
Claude Opus 5.5.

| Scenario | Result |
|---|---|
| Reply appears in pieces while it is generated | Passed |
| Message box is locked while a reply is generated | Passed |
| A later turn recalls a name given in an earlier turn | Passed |
| Web search shows its status and lists source links | Passed |
| A follow-up question works after a search | Passed |
| The model reported by the API appears under the reply | Passed |
| Cost is shown and a new conversation resets it | Passed |
| A low budget blocks the request before it is sent | Passed |
| Missing or rejected API key shows a clear message | Passed |
| The key does not appear in the page or in WebSocket traffic | Passed |

## Direct checks against the API

All three models answered through `chat.py` and kept context across turns.
With a long system prompt, the second turn read 16,080 tokens from the prompt
cache. Web search worked on all three models, and the second turn reused the
earlier results without a new search.

| Model | Two turns, no search | Two turns, one search |
|---|---|---|
| Claude Opus 5.5 | $0.00078 | $0.0725 |
| Claude Sonnet 5.5 | $0.00048 | $0.0397 |
| Claude Haiku 5.5 | $0.00002 | $0.0116 |

These are estimates at public list prices for short test messages. The
default budget is USD 1.00 because the app reserves the largest possible cost
of the next reply before sending it. That is about USD 0.35 on Opus 5.5, so a
lower default stopped a conversation after two searches.

## Problems found during testing

1. **Error messages disappeared.** The rerun at the end of a turn removed them.
   They are now stored and shown on the next render.
2. **A message sent as a reply finished was lost.** The rerun discarded it. The
   message is now held as pending and the box is locked until the reply ends.

## Not verified

- A real refusal or a real fallback to another model. These cannot be triggered
  on demand, so refusal handling is covered only by mocked tests.
- A real `pause_turn` from a long search. Mocked tests only.
- Image neutralization against a real reply. Unit tests only.
- The browser tests on Sonnet 5.5 and Haiku 5.5, and on Python 3.12.

Web results can differ between sources, so check the source links for anything
important.
