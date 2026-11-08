"""Local chatbot demo on the Claude API. Run with: streamlit run app.py"""

import os

import anthropic
import streamlit as st

import chat

DEFAULT_SYSTEM = "You are a helpful assistant in a demo chat. Answer clearly and briefly."

st.set_page_config(page_title="Claude Chat Demo", page_icon="💬")
st.title("Claude Chat Demo")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not set. Export it in the shell that starts the app.")
    st.stop()


def reset() -> None:
    st.session_state.display = []  # what the page shows
    st.session_state.history = []  # what the API receives, with full content blocks
    st.session_state.spent = 0.0
    st.session_state.notice = None  # survives the rerun at the end of a turn
    st.session_state.pending = None  # message being answered; locks the input


def submit() -> None:
    st.session_state.pending = st.session_state.chat_box


if "history" not in st.session_state:
    reset()


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


with st.sidebar:
    model_id = st.selectbox(
        "Model",
        list(chat.MODELS),
        index=list(chat.MODELS).index(chat.DEFAULT_MODEL),
        format_func=lambda m: chat.MODELS[m].label,
        on_change=reset,
    )
    system = st.text_area("System prompt", DEFAULT_SYSTEM, height=120)
    budget = st.number_input("Budget per conversation (USD)", 0.05, 5.0, chat.DEFAULT_BUDGET_USD, 0.05)
    st.button("New conversation", on_click=reset, use_container_width=True)
    st.metric("Spent in this conversation (estimate)", f"${st.session_state.spent:.4f}")
    turns = len(st.session_state.display) // 2
    st.caption(f"Turn {turns} of {chat.MAX_TURNS}")

for message in st.session_state.display:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for title, url in message.get("sources", []):
            st.markdown(f"- [{title}]({url})")
        if "model" in message:
            st.caption(f"Answered by {message['model']}")

if st.session_state.notice:
    st.warning(st.session_state.notice)
    st.session_state.notice = None

st.chat_input(
    "Write a message",
    key="chat_box",
    max_chars=chat.MAX_INPUT_CHARS,
    on_submit=submit,
    disabled=st.session_state.pending is not None,
)
prompt = st.session_state.pending
if not prompt:
    st.stop()

blocked = None
if turns >= chat.MAX_TURNS:
    blocked = "This conversation reached its turn limit. Start a new one."
elif st.session_state.spent + chat.max_turn_cost(model_id) > budget:
    blocked = "The next reply could go over the budget. Start a new conversation or raise the budget."
if blocked:
    st.session_state.notice = blocked
    st.session_state.pending = None
    st.rerun()

with st.chat_message("user"):
    st.markdown(prompt)

error = None
text = ""
with st.chat_message("assistant"):
    status = st.empty()
    reply = chat.ReplyStream(
        get_client(),
        model_id,
        system,
        [*st.session_state.history, {"role": "user", "content": prompt}],
        on_search=lambda: status.caption("Searching the web..."),
    )
    try:
        text = st.write_stream(reply)
    except anthropic.AuthenticationError:
        error = "The API key was rejected. Check ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        error = "Rate limit reached. Wait a moment and try again."
    except anthropic.APIConnectionError:
        error = "Could not reach the Claude API. Check your connection."
    except anthropic.APIStatusError as exc:
        error = f"The API returned an error ({exc.status_code})."

    status.empty()
    if error is None and reply.final.stop_reason == "refusal":
        error = "Claude declined this request. Try asking in a different way."
    if error is None and reply.final.stop_reason == "pause_turn":
        error = "The search took too long. Try a narrower question."
    if error is None and reply.final.stop_reason == "max_tokens":
        st.session_state.notice = "The reply reached the length limit."
    if error:
        st.session_state.notice = error

st.session_state.spent += reply.cost()

# A failed or declined turn stays out of the history, so the next request is valid.
if error is None:
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.extend(reply.assistant_turns)
    st.session_state.display.append({"role": "user", "content": prompt})
    # The model the API reports for this reply, which differs after a fallback.
    st.session_state.display.append(
        {"role": "assistant", "content": text, "model": reply.final.model, "sources": reply.sources()}
    )

st.session_state.pending = None
st.rerun()
