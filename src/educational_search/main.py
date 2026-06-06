"""
Educational Content Search Engine
Main application entry point.
"""

import os

import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from botocore.exceptions import PartialCredentialsError
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
RESPONSE_FORMATS = {
    "Document table": (
        "Create a Markdown table with exactly 3 columns: Document Title, "
        "AI Summary, and Website Link."
    ),
    "Concise summary": (
        "Write a concise, well-organized summary with the most relevant "
        "documents and links."
    ),
    "Bullet list": (
        "Write a scannable bullet list. Include document titles, short "
        "summaries, and links when available."
    ),
    "Links only": (
        "Return only the most relevant document titles and links, with one "
        "short reason each."
    ),
}


def get_aws_region():
    return os.getenv("AWS_DEFAULT_REGION", "us-west-2")


def get_bedrock_model_arn():
    model_id = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
    if model_id.startswith("arn:"):
        return model_id

    return f"arn:aws:bedrock:{get_aws_region()}::foundation-model/{model_id}"


def build_search_prompt(user_question, response_format):
    format_instruction = RESPONSE_FORMATS[response_format]

    return f"""
Situation
You are working with a collection of documents that need to be organized and
presented in a structured format for easy reference and access.

Task
You are an expert data analyst and information organizer. Answer the user's
question using the retrieved educational documents.

Objective
Provide a clear, organized overview of multiple documents that allows for quick
scanning of content and easy access to source materials.

Response format
{format_instruction}

Quality rules
Make summaries comprehensive enough to understand each document's purpose and
content without being overly lengthy. Format links properly. If information is
missing for a document, clearly indicate that rather than leaving it blank.
Organize multiple documents in a logical order: relevance, alphabetical by title,
or chronological.

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


def get_location_url(reference):
    location = reference.get("location", {})

    for value in location.values():
        if isinstance(value, dict):
            for key in ("uri", "url"):
                if value.get(key):
                    return value[key]

    metadata = reference.get("metadata", {})
    for key in ("source_url", "url", "uri", "x-amz-bedrock-kb-source-uri"):
        if metadata.get(key):
            return metadata[key]

    return ""


def get_source_title(reference, fallback_index):
    metadata = reference.get("metadata", {})
    for key in ("title", "document_title", "DocumentTitle", "source"):
        if metadata.get(key):
            return metadata[key]

    url = get_location_url(reference)
    if url:
        return url.rsplit("/", 1)[-1] or url

    return f"Source {fallback_index}"


def get_reference_snippet(reference, max_length=220):
    content = reference.get("content", {})
    text = content.get("text", "") if isinstance(content, dict) else ""
    text = " ".join(text.split())

    if len(text) <= max_length:
        return text

    return f"{text[:max_length].rstrip()}..."


def extract_source_citations(response):
    sources = []
    seen = set()

    for citation in response.get("citations", []):
        for reference in citation.get("retrievedReferences", []):
            url = get_location_url(reference)
            snippet = get_reference_snippet(reference)
            dedupe_key = (url, snippet)

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            sources.append(
                {
                    "title": get_source_title(reference, len(sources) + 1),
                    "url": url,
                    "snippet": snippet,
                }
            )

    return sources


def build_sources_markdown(sources):
    if not sources:
        return ""

    lines = ["", "### Sources"]
    for index, source in enumerate(sources, start=1):
        title = source["title"]
        url = source["url"]
        snippet = source["snippet"]
        link = f"[{title}]({url})" if url else title
        lines.append(f"{index}. {link}")

        if snippet:
            lines.append(f"   {snippet}")

    return "\n".join(lines)


def get_friendly_aws_error(exc):
    if isinstance(exc, (NoCredentialsError, PartialCredentialsError)):
        return (
            "AWS credentials are missing or incomplete. Check AWS_ACCESS_KEY_ID "
            "and AWS_SECRET_ACCESS_KEY in your .env file."
        )

    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = error.get("Code", "")

        if code in {"AccessDeniedException", "AccessDenied", "UnauthorizedException"}:
            return (
                "AWS denied this request. Check that your credentials can access "
                "Bedrock, the Knowledge Base, and the selected model."
            )

        if code in {"ResourceNotFoundException", "ValidationException"}:
            return (
                "AWS could not find or use one of the configured resources. Check "
                "your KNOWLEDGE_BASE_ID, AWS_DEFAULT_REGION, and BEDROCK_MODEL_ID."
            )

        if code in {"ExpiredToken", "UnrecognizedClientException"}:
            return (
                "Your AWS credentials look expired or invalid. Refresh them and "
                "try again."
            )

        if code in {"ThrottlingException", "TooManyRequestsException"}:
            return "AWS is rate limiting requests right now. Wait a moment and retry."

    if isinstance(exc, BotoCoreError):
        return (
            "The AWS client had trouble making the request. Check your network, "
            "region, and credentials."
        )

    return "Something went wrong while searching your documents."


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
        st.error(get_friendly_aws_error(exc))
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
    response_format = st.selectbox(
        "Response format",
        list(RESPONSE_FORMATS.keys()),
        index=0,
    )
    result_count = st.slider(
        "Documents to search",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
    )
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
                    input={"text": build_search_prompt(prompt, response_format)},
                    retrieveAndGenerateConfiguration={
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration": {
                            "knowledgeBaseId": kb_id,
                            "modelArn": get_bedrock_model_arn(),
                            "retrievalConfiguration": {
                                "vectorSearchConfiguration": {
                                    "numberOfResults": result_count,
                                },
                            },
                        },
                    },
                )

                answer = response["output"]["text"]
                sources_markdown = build_sources_markdown(
                    extract_source_citations(response)
                )
                full_answer = f"{answer}{sources_markdown}"

                st.markdown(full_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer}
                )
            except Exception as exc:
                st.error(get_friendly_aws_error(exc))
                with st.expander("Technical details"):
                    st.exception(exc)
