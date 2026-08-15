"""Streamlit chat interface for the agentic resume screening system.

Two-column layout: chat on the left, live agent state (round, requirements,
shortlist) in the sidebar. An expander below the chat input shows which
graph nodes ran and which tools they called this turn, so the agent's
reasoning is visible rather than opaque. All conversation-driving logic
lives in matching_agent.run_turn — this file only renders state.
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st

# Import resume_rag before chromadb: it patches sys.modules["sqlite3"] with
# pysqlite3-binary (chromadb needs sqlite3 >= 3.35.0) before anything here
# imports chromadb directly. See resume_rag.py / job_matcher.py for the
# same ordering requirement.
from resume_rag import CHROMA_PERSIST_DIR, COLLECTION_NAME

import chromadb

from matching_agent import build_graph, run_turn

st.set_page_config(page_title="Agentic Profile Matching", layout="wide")


@st.cache_resource
def get_app():
    return build_graph()


def _corpus_is_empty() -> bool:
    """Check the resume collection without paying for an embedding-model load."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return collection.count() == 0


if "thread_id" not in st.session_state:
    # Unique per browser session — get_app() is cache_resource, so the
    # compiled graph (and its in-memory MemorySaver) is a single object
    # shared by every session on this server. A fixed thread_id here would
    # mean two visitors silently share one conversation.
    st.session_state.thread_id = str(uuid.uuid4())
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

app = get_app()
config = {"configurable": {"thread_id": st.session_state.thread_id}}
corpus_empty = _corpus_is_empty()

with st.sidebar:
    st.header("Agent state")
    snapshot = app.get_state(config)
    state = snapshot.values if snapshot.values else {}

    st.metric("Round", state.get("round_number", 1))

    st.subheader("Requirements")
    requirements = state.get("requirements", {})
    if requirements:
        st.write(f"**Role:** {requirements.get('role_title', 'n/a')}")
        must_have = requirements.get("must_have", {})
        st.write(f"**Must-have skills:** {', '.join(must_have.get('skills', [])) or 'none'}")
        st.write(f"**Min years:** {must_have.get('min_years', 0)}")
        nice_to_have = requirements.get("nice_to_have", {})
        st.write(f"**Nice-to-have skills:** {', '.join(nice_to_have.get('skills', [])) or 'none'}")
    else:
        st.caption("No requirements extracted yet.")

    st.subheader("Shortlist")
    shortlist = state.get("shortlist", [])
    if shortlist:
        st.dataframe(
            [
                {
                    "Candidate": c["candidate_name"],
                    "Score": c["match_score"],
                    "Matched skills": ", ".join(c.get("matched_skills", [])),
                }
                for c in shortlist
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No shortlist yet.")

st.title("Agentic Profile Matching")
st.caption("Screen resumes against a job description through a multi-round, conversational workflow.")

if corpus_empty:
    st.error(
        "No resumes are indexed yet. Run `python generate_sample_data.py` "
        "(if you haven't already) and then `python resume_rag.py` to ingest "
        "the corpus, then reload this page."
    )

for message in st.session_state.display_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "e.g. Screen candidates against data/job_descriptions/senior_ml_engineer.txt",
    disabled=corpus_empty,
)

if user_input:
    st.session_state.display_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            turn = run_turn(app, config, user_input)

        # Render every AIMessage the graph produced since the user's latest
        # message (a turn can emit more than one — e.g. a ranking-delta
        # narrative followed by the updated report).
        messages = turn["state"]["messages"]
        last_human_idx = max(
            i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
        )
        turn_ai_messages = [m for m in messages[last_human_idx + 1 :] if isinstance(m, AIMessage)]

        rendered = "\n\n---\n\n".join(m.content for m in turn_ai_messages) or (
            "(No report generated this turn — see the reasoning trace below.)"
        )
        st.markdown(rendered)

    st.session_state.display_history.append({"role": "assistant", "content": rendered})
    st.session_state.last_trace = turn["trace"]
    st.rerun()

with st.expander("Agent reasoning trace", expanded=False):
    if st.session_state.last_trace:
        for step in st.session_state.last_trace:
            tools = ", ".join(step["tools_called"]) if step["tools_called"] else "—"
            st.write(f"**{step['node']}** — tools called: {tools}")
    else:
        st.caption("No turns run yet.")
