"""
Educational Content Search Engine
Main application entry point.
"""

import os

import boto3
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


def get_aws_region():
    return os.getenv("AWS_DEFAULT_REGION", "us-west-2")


def get_bedrock_model_arn():
    model_id = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
    if model_id.startswith("arn:"):
        return model_id

    return f"arn:aws:bedrock:{get_aws_region()}::foundation-model/{model_id}"


def build_search_prompt(user_question):
    return f"""
Situation
You are working with a collection of documents that need to be organized and
presented in a structured format for easy reference and access.

Task
You are an expert data analyst and information organizer. Create a comprehensive
table with exactly 3 columns that organizes the provided documents. Format the
output as a clean, well-structured Markdown table where each row represents one
document.

Objective
Provide a clear, organized overview of multiple documents that allows for quick
scanning of content and easy access to source materials.

Knowledge
The table must contain exactly these 3 columns:

Document Title - The official or descriptive title of each document
AI Summary - A concise, informative summary that captures the key points, main
topics, and essential information from each document
Website Link - The complete URL where the document can be accessed online

Make summaries comprehensive enough to understand the document's purpose and
content without being overly lengthy. Format links properly. If information is
missing for a document, clearly indicate that in the appropriate cell rather than
leaving it blank.

When multiple documents are provided, organize them in a logical order:
alphabetical by title, chronological, or by relevance.

User question:
{user_question}
"""


def get_missing_environment_variables():
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "KNOWLEDGE_BASE_ID"]
    return [
        var
        for var in required_vars
        if not os.getenv(var) or os.getenv(var) == f"your_{var.lower()}_here"
    ]


@st.cache_resource
def setup_bedrock():
    try:
        client_kwargs = {
            "service_name": "bedrock-agent-runtime",
            "region_name": os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
        }

        if os.getenv("AWS_ACCESS_KEY_ID"):
            client_kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
            client_kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")

            if os.getenv("AWS_SESSION_TOKEN"):
                client_kwargs["aws_session_token"] = os.getenv("AWS_SESSION_TOKEN")

        return boto3.client(**client_kwargs)
    except Exception as exc:
        st.error("Failed to set up the AWS Bedrock client.")
        with st.expander("Technical details"):
            st.exception(exc)
        return None


st.set_page_config(
    page_title="Ou Search",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #4f46e5 0%, #0f766e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .search-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4f46e5;
    }
    .stChatMessage {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        line-height: 1.6;
        font-size: 16px;
    }
    .stChatMessage p {
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .stChatMessage table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 14px;
    }
    .stChatMessage th {
        background: #4f46e5;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    .stChatMessage td {
        padding: 12px;
        border-bottom: 1px solid #e9ecef;
        vertical-align: top;
        color: #2c3e50;
    }
    .stChatMessage tr:nth-child(even) {
        background: #f8f9fa;
    }
    .stChatMessage a {
        color: #4f46e5;
        text-decoration: none;
    }
    .stChatMessage a:hover {
        text-decoration: underline;
    }
    .stSpinner {
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1>Ou Search</h1>
    <p>Your intelligent Asian American educational content search engine</p>
</div>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    st.caption(f"Region: {get_aws_region()}")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

missing_vars = get_missing_environment_variables()

if missing_vars:
    st.error(f"Missing configuration: {', '.join(missing_vars)}")
    st.info(
        "Create a .env file from .env.example and add your AWS credentials, "
        "Knowledge Base ID, and AWS region."
    )
    st.stop()

bedrock = setup_bedrock()
kb_id = os.getenv("KNOWLEDGE_BASE_ID")

if not bedrock:
    st.error("AWS connection failed. Unable to connect to Bedrock.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown('<div class="search-container">', unsafe_allow_html=True)
st.markdown("### Ask Your Question")
st.markdown("*Search through educational documents and get organized answers.*")
st.markdown("</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask me anything related to your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = bedrock.retrieve_and_generate(
                    input={"text": build_search_prompt(prompt)},
                    retrieveAndGenerateConfiguration={
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration": {
                            "knowledgeBaseId": kb_id,
                            "modelArn": get_bedrock_model_arn(),
                        },
                    },
                )

                answer = response["output"]["text"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:
                st.error("Something went wrong while searching your documents.")
                with st.expander("Technical details"):
                    st.exception(exc)
