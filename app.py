import os
import inspect
import shutil
import mimetypes
import smtplib
from pathlib import Path
from email.message import EmailMessage

import pandas as pd
import streamlit as st

from gradio_client import Client

from src.document_processing.document_loader import PosterDocumentLoader
from src.vision.vision_model import LocalVisionModel
from src.extraction.metadata_extractor import MetadataExtractor
from src.extraction.schemas import PosterMetadata
from src.rag.rag_pipeline import PosterRAG
from src.summarization.summarizer import PosterSummarizer
from src.student_filtering.student_filter import StudentFilter
from src.email.email_generator import EmailGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered Intelligent University Notice Distribution System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    html,
    body,
    [class*="css"] {
        font-size: 19px !important;
    }

    .stApp {
        background-color: #f7f8fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1600px;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
    }

    h2 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
    }

    h3 {
        font-size: 2rem !important;
        font-weight: 750 !important;
    }

    h4 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    p,
    li,
    label,
    .stMarkdown,
    .stText,
    .stCaption {
        font-size: 1.15rem !important;
    }

    .dashboard-title {
        font-size: 3rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #172554;
    }

    .dashboard-subtitle {
        font-size: 1.3rem !important;
        color: #4b5563;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #dbe1ea;
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 160px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .metric-label {
        color: #4b5563;
        font-size: 1rem !important;
        font-weight: 700;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 2.7rem !important;
        font-weight: 800;
        margin-top: 0.5rem;
        color: #111827;
    }

    .metric-caption {
        color: #6b7280;
        font-size: 1rem !important;
        margin-top: 0.5rem;
    }

    .info-card {
        background: white;
        border: 1px solid #dbe1ea;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .stButton > button {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        min-height: 3.2rem;
        border-radius: 10px;
    }

    input,
    textarea {
        font-size: 1.1rem !important;
    }

    [data-testid="stDataFrame"] {
        font-size: 1.05rem !important;
    }

    [data-testid="stExpander"] {
        font-size: 1.1rem !important;
    }

    [data-testid="stSidebar"] {
        font-size: 1.05rem !important;
    }

    .email-preview {
        background: white;
        border: 1px solid #dbe1ea;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .email-header {
        font-size: 1.2rem !important;
        font-weight: 700;
        color: #172554;
        margin-bottom: 0.75rem;
    }

    .footer-text {
        font-size: 1rem !important;
        color: #6b7280;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "processed": False,

    "poster_path": None,
    "poster_name": None,

    "generated_image_path": None,
    "generated_image_name": None,

    "students_df": None,
    "filtered_students": None,

    "metadata": None,
    "summary": None,

    "rag_answers": [],
    "rag_context": "",

    "generated_emails": [],

    "sender_email": "",
    "sender_password": "",

    "email_send_confirmed": False,
    "email_send_results": [],
    "emails_sent": False,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


# ============================================================
# FORGE CONFIGURATION
# ============================================================

FORGE_URL = "http://127.0.0.1:7860"

FORGE_ENDPOINT = "/txt2img"

GENERATED_IMAGE_DIRECTORY = Path(
    "data/generated_invitations"
)

GENERATED_IMAGE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BACKEND INITIALIZATION
# ============================================================

@st.cache_resource
def get_document_loader():

    return PosterDocumentLoader(
        languages=["en"],
        gpu=False,
    )


# ============================================================
# IMPORTANT:
# Gemma 3 Vision is NOT loaded.
#
# Your previous error:
#
# ResponseError:
# model requires more system memory
# (4.3 GiB) than is available (2.2 GiB)
#
# Therefore the vision model is deliberately disabled.
# ============================================================

@st.cache_resource
def get_vision_model():

    return None


@st.cache_resource
def get_metadata_extractor():

    return MetadataExtractor(
        model_name="gemma3:latest",
    )


@st.cache_resource
def get_rag():

    return PosterRAG(
        model_name="gemma3:latest",
    )


@st.cache_resource
def get_summarizer():

    rag = get_rag()

    return PosterSummarizer(
        rag_pipeline=rag,
        model_name="gemma3:latest",
    )


@st.cache_resource
def get_email_generator():

    return EmailGenerator(
        model_name="gemma3:latest",
    )


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(value):
    """
    Converts metadata values safely to readable text.
    """

    if value is None:

        return ""

    try:

        if pd.isna(value):

            return ""

    except Exception:

        pass

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return ", ".join(
            str(item)
            for item in value
            if item is not None
        )

    if isinstance(
        value,
        dict,
    ):

        return ", ".join(
            f"{key}: {value}"
            for key, value in value.items()
        )

    return str(value)


# ============================================================
# METADATA NORMALIZATION
# ============================================================

def normalize_metadata_for_pydantic(
    raw_metadata
):

    if raw_metadata is None:

        return {}

    if not isinstance(
        raw_metadata,
        dict,
    ):

        try:

            raw_metadata = dict(
                raw_metadata
            )

        except Exception:

            return {}

    cleaned = dict(
        raw_metadata
    )

    # --------------------------------------------------------
    # LIST FIELDS
    # --------------------------------------------------------

    list_fields = [
        "important_dates",
        "keywords",
        "guest_speakers",
        "contact_phone",
    ]

    for field in list_fields:

        if field not in cleaned:

            continue

        value = cleaned[field]

        if value is None:

            cleaned[field] = []

        elif isinstance(
            value,
            str,
        ):

            value = value.strip()

            if not value:

                cleaned[field] = []

            elif value.lower() in {
                "not available",
                "not provided",
                "none",
                "n/a",
                "na",
                "unknown",
                "default",
                "null",
            }:

                cleaned[field] = []

            else:

                cleaned[field] = [
                    value
                ]

        elif isinstance(
            value,
            tuple,
        ):

            cleaned[field] = list(
                value
            )

        elif isinstance(
            value,
            set,
        ):

            cleaned[field] = list(
                value
            )

        elif not isinstance(
            value,
            list,
        ):

            cleaned[field] = [
                str(value)
            ]

    # --------------------------------------------------------
    # TARGET AUDIENCE TYPE
    # --------------------------------------------------------

    if "target_audience_type" in cleaned:

        value = cleaned[
            "target_audience_type"
        ]

        if value is None:

            cleaned[
                "target_audience_type"
            ] = "Inferred"

        elif isinstance(
            value,
            str,
        ):

            value = value.strip()

            if value not in {
                "Explicit",
                "Inferred",
            }:

                cleaned[
                    "target_audience_type"
                ] = "Inferred"

    # --------------------------------------------------------
    # STRING FIELDS
    # --------------------------------------------------------

    string_fields = [
        "poster_type",
        "title",
        "department",
        "university",
        "description",
        "short_summary",
        "event_date",
        "venue",
        "registration_deadline",
        "registration_link",
        "contact_person",
        "email",
        "eligibility",
        "target_audience",
        "other_information",
    ]

    for field in string_fields:

        if field not in cleaned:

            continue

        value = cleaned[field]

        if value is None:

            cleaned[field] = ""

        elif isinstance(
            value,
            list,
        ):

            cleaned[field] = ", ".join(
                str(item)
                for item in value
                if item is not None
            )

        elif isinstance(
            value,
            dict,
        ):

            cleaned[field] = ", ".join(
                f"{key}: {value}"
                for key, value in value.items()
            )

        else:

            cleaned[field] = str(value)

    return cleaned


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

def save_uploaded_file(
    uploaded_file
):

    upload_directory = Path(
        "data/uploads"
    )

    upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = Path(
        uploaded_file.name
    ).name

    target_path = (
        upload_directory
        / filename
    )

    with open(
        target_path,
        "wb",
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return str(
        target_path
    )


# ============================================================
# OCR
# ============================================================

def extract_ocr_text(
    loader,
    file_path,
):

    documents = (
        loader.load_single_document(
            file_path
        )
    )

    text_parts = []

    for document in documents:

        page_content = getattr(
            document,
            "page_content",
            "",
        )

        if page_content:

            text_parts.append(
                page_content
            )

    return "\n\n".join(
        text_parts
    ).strip()


# ============================================================
# VISION
# ============================================================

def run_vision_analysis(
    vision_model,
    file_path,
):

    """
    Vision analysis intentionally disabled.

    Reason:
    Gemma 3 Vision previously required approximately
    4.3 GiB of system memory while only approximately
    2.2 GiB was available.

    OCR remains the primary source.
    """

    return (
        "Vision analysis was intentionally skipped "
        "because the local Gemma 3 vision model requires "
        "more system memory than is currently available. "
        "Use OCR/document extraction as the primary source."
    )


# ============================================================
# FORGE ARGUMENT BUILDER
# ============================================================

def build_forge_arguments(
    prompt,
    negative_prompt,
):
    """
    Builds the exact /txt2img parameter list based on the
    working Forge endpoint discovered from forge_info.txt.

    The confirmed endpoint has 123 parameters.

    The two important parameters are:

        index 1 = Prompt
        index 2 = Negative prompt

    The remaining values use the same defaults that worked
    in forge_test.py.
    """

    args = [None] * 123

    # --------------------------------------------------------
    # 0
    # parameter_47 / id_task
    # --------------------------------------------------------

    args[0] = None

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    args[1] = prompt

    # --------------------------------------------------------
    # NEGATIVE PROMPT
    # --------------------------------------------------------

    args[2] = negative_prompt

    # --------------------------------------------------------
    # Styles
    # --------------------------------------------------------

    args[3] = []

    # --------------------------------------------------------
    # Batch count
    # --------------------------------------------------------

    args[4] = 1

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    args[5] = 1

    # --------------------------------------------------------
    # CFG
    # --------------------------------------------------------

    args[6] = 7.0

    # --------------------------------------------------------
    # Distilled CFG
    # --------------------------------------------------------

    args[7] = 3.5

    # --------------------------------------------------------
    # Height
    # --------------------------------------------------------

    args[8] = 512

    # --------------------------------------------------------
    # Width
    # --------------------------------------------------------

    args[9] = 512

    # --------------------------------------------------------
    # Hires Fix
    # --------------------------------------------------------

    args[10] = False

    # --------------------------------------------------------
    # Denoising strength
    # --------------------------------------------------------

    args[11] = 0.7

    # --------------------------------------------------------
    # Upscale by
    # --------------------------------------------------------

    args[12] = 2.0

    # --------------------------------------------------------
    # Upscaler
    # --------------------------------------------------------

    args[13] = "Latent"

    # --------------------------------------------------------
    # Hires steps
    # --------------------------------------------------------

    args[14] = 0

    # --------------------------------------------------------
    # Resize width
    # --------------------------------------------------------

    args[15] = 0

    # --------------------------------------------------------
    # Resize height
    # --------------------------------------------------------

    args[16] = 0

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    args[17] = "Use same checkpoint"

    # --------------------------------------------------------
    # Hires sampling method
    # --------------------------------------------------------

    args[18] = "Use same sampler"

    # --------------------------------------------------------
    # Hires schedule
    # --------------------------------------------------------

    args[19] = "Use same scheduler"

    # --------------------------------------------------------
    # Hires prompt
    # --------------------------------------------------------

    args[20] = ""

    # --------------------------------------------------------
    # Hires negative
    # --------------------------------------------------------

    args[21] = ""

    # --------------------------------------------------------
    # Override settings
    # --------------------------------------------------------

    args[22] = None

    # --------------------------------------------------------
    # Script
    # --------------------------------------------------------

    args[23] = None

    # --------------------------------------------------------
    # Sampling steps
    # --------------------------------------------------------

    args[24] = 20

    # --------------------------------------------------------
    # Sampling method
    # --------------------------------------------------------

    args[25] = "DPM++ 2M"

    # --------------------------------------------------------
    # Schedule type
    # --------------------------------------------------------

    args[26] = "Automatic"

    # --------------------------------------------------------
    # Refiner
    # --------------------------------------------------------

    args[27] = False

    # --------------------------------------------------------
    # Refiner checkpoint
    # --------------------------------------------------------

    args[28] = ""

    # --------------------------------------------------------
    # Switch at
    # --------------------------------------------------------

    args[29] = 0.8

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    args[30] = -1.0

    # --------------------------------------------------------
    # Extra
    # --------------------------------------------------------

    args[31] = False

    # --------------------------------------------------------
    # Variation seed
    # --------------------------------------------------------

    args[32] = -1.0

    # --------------------------------------------------------
    # Variation strength
    # --------------------------------------------------------

    args[33] = 0.0

    # --------------------------------------------------------
    # Resize seed width
    # --------------------------------------------------------

    args[34] = 0

    # --------------------------------------------------------
    # Resize seed height
    # --------------------------------------------------------

    args[35] = 0

    # --------------------------------------------------------
    # Remaining Forge extension defaults
    # --------------------------------------------------------

    args[36] = False
    args[37] = 7.0
    args[38] = 1.0
    args[39] = "Constant"
    args[40] = 0.0
    args[41] = "Constant"
    args[42] = 0.0
    args[43] = 1.0
    args[44] = "enable"
    args[45] = "MEAN"
    args[46] = "AD"
    args[47] = 1.0

    args[48] = False
    args[49] = 1.01
    args[50] = 1.02
    args[51] = 0.99
    args[52] = 0.95

    args[53] = False
    args[54] = 0.5
    args[55] = 2.0

    args[56] = False
    args[57] = 3.0

    args[58] = False
    args[59] = 3
    args[60] = 2.0
    args[61] = 0.0
    args[62] = 0.35
    args[63] = True
    args[64] = "bicubic"
    args[65] = "bicubic"

    args[66] = False
    args[67] = 0.0
    args[68] = "anisotropic"
    args[69] = 0.0
    args[70] = "reinhard"
    args[71] = 100.0
    args[72] = 0.0
    args[73] = "subtract"
    args[74] = 0.0
    args[75] = 0.0
    args[76] = "gaussian"
    args[77] = "add"
    args[78] = 0.0
    args[79] = 100
    args[80] = 127
    args[81] = 0.0
    args[82] = "hard_clamp"
    args[83] = 5.0
    args[84] = 0.0
    args[85] = None
    args[86] = None

    args[87] = False
    args[88] = "MultiDiffusion"
    args[89] = 768
    args[90] = 768
    args[91] = 64
    args[92] = 4
    args[93] = False
    args[94] = False
    args[95] = False

    args[96] = False
    args[97] = False
    args[98] = "positive"
    args[99] = "comma"
    args[100] = 0
    args[101] = False
    args[102] = False
    args[103] = "start"
    args[104] = ""

    args[105] = "Seed"
    args[106] = ""
    args[107] = ""

    args[108] = "Nothing"
    args[109] = ""
    args[110] = ""

    args[111] = "Nothing"
    args[112] = ""
    args[113] = ""

    args[114] = True
    args[115] = False
    args[116] = False
    args[117] = False
    args[118] = False
    args[119] = False
    args[120] = False
    args[121] = 0
    args[122] = False

    return args


# ============================================================
# FORGE IMAGE GENERATION
# ============================================================

def generate_invitation_image(
    prompt,
    negative_prompt,
):

    """
    Calls the locally running Forge WebUI.

    Forge:
        http://127.0.0.1:7860

    Endpoint:
        /txt2img
    """

    try:

        client = Client(
            FORGE_URL
        )

        forge_arguments = (
            build_forge_arguments(
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
        )

        result = client.predict(
            *forge_arguments,
            api_name=FORGE_ENDPOINT,
        )

        if not result:

            raise RuntimeError(
                "Forge returned an empty response."
            )

        images = result[0]

        if not images:

            raise RuntimeError(
                "Forge completed but returned no image."
            )

        first_image = images[0]

        if isinstance(
            first_image,
            dict,
        ):

            image_path = first_image.get(
                "image"
            )

        else:

            image_path = str(
                first_image
            )

        if not image_path:

            raise RuntimeError(
                "Forge returned an invalid image path."
            )

        source_path = Path(
            image_path
        )

        if not source_path.exists():

            raise FileNotFoundError(
                "Generated Forge image was not found: "
                f"{source_path}"
            )

        output_path = (
            GENERATED_IMAGE_DIRECTORY
            / "generated_invitation.png"
        )

        shutil.copy2(
            source_path,
            output_path,
        )

        return str(
            output_path
        )

    except Exception as error:

        raise RuntimeError(
            "Forge image generation failed: "
            f"{type(error).__name__}: {error}"
        ) from error


# ============================================================
# METADATA DATAFRAME
# ============================================================

def metadata_dataframe(
    metadata
):

    fields = [
        ("Poster Type", "poster_type"),
        ("Title", "title"),
        ("Department", "department"),
        ("University", "university"),
        ("Description", "description"),
        ("Short Summary", "short_summary"),
        ("Event Date", "event_date"),
        ("Important Dates", "important_dates"),
        ("Venue", "venue"),
        ("Registration Deadline", "registration_deadline"),
        ("Registration Link", "registration_link"),
        ("Guest Speakers", "guest_speakers"),
        ("Contact Person", "contact_person"),
        ("Contact Phone", "contact_phone"),
        ("Email", "email"),
        ("Eligibility", "eligibility"),
        ("Target Audience", "target_audience"),
        ("Target Audience Type", "target_audience_type"),
        ("Keywords", "keywords"),
        ("Other Information", "other_information"),
    ]

    rows = []

    for label, attribute in fields:

        value = getattr(
            metadata,
            attribute,
            "",
        )

        rows.append(
            (
                label,
                safe_text(value),
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Field",
            "Value",
        ],
    )


# ============================================================
# RAG CONTEXT
# ============================================================

def create_rag_context(
    answers
):

    if not answers:

        return ""

    context_parts = []

    for index, item in enumerate(
        answers,
        start=1,
    ):

        context_parts.append(
            (
                f"QUESTION {index}:\n"
                f"{item['question']}\n\n"
                f"ANSWER {index}:\n"
                f"{item['answer']}"
            )
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(
    label,
    value,
    caption,
):

    return f"""
    <div class="metric-card">

        <div class="metric-label">
            {label}
        </div>

        <div class="metric-value">
            {value}
        </div>

        <div class="metric-caption">
            {caption}
        </div>

    </div>
    """


# ============================================================
# RESET
# ============================================================

def reset_application():

    for key, value in DEFAULT_STATE.items():

        st.session_state[key] = value


# ============================================================
# EMAIL GENERATOR COMPATIBILITY
# ============================================================

def call_email_generator(
    generator,
    metadata,
    summary,
    student,
    rag_context,
):

    method = (
        generator.generate_email
    )

    try:

        signature = inspect.signature(
            method
        )

        parameters = (
            signature.parameters
        )

        kwargs = {}

        possible_arguments = {

            "metadata": metadata,

            "summary": summary,

            "student": student,

            "rag_context": rag_context,

        }

        for name, value in (
            possible_arguments.items()
        ):

            if name in parameters:

                kwargs[name] = value

        return method(
            **kwargs
        )

    except Exception:

        return method(
            metadata=metadata,
            summary=summary,
            student=student,
            rag_context=rag_context,
        )


# ============================================================
# NORMALIZE EMAIL
# ============================================================

def normalize_email_result(
    email_result
):

    if isinstance(
        email_result,
        dict,
    ):

        subject = email_result.get(
            "subject",
            "Academic Opportunity",
        )

        body = email_result.get(
            "body",
            "",
        )

        return {
            "subject": str(subject),
            "body": str(body),
        }

    if isinstance(
        email_result,
        str,
    ):

        return {
            "subject": "Academic Opportunity",
            "body": email_result,
        }

    subject = getattr(
        email_result,
        "subject",
        "Academic Opportunity",
    )

    body = getattr(
        email_result,
        "body",
        str(email_result),
    )

    return {
        "subject": str(subject),
        "body": str(body),
    }


# ============================================================
# DIRECT GMAIL SMTP SENDER
# ============================================================

def send_email_with_attachments(
    sender_email,
    sender_password,
    recipient_email,
    recipient_name,
    subject,
    body,
    attachment_paths,
):

    if not sender_email:

        raise ValueError(
            "Sender email is missing."
        )

    if not sender_password:

        raise ValueError(
            "Sender App Password is missing."
        )

    if not recipient_email:

        raise ValueError(
            "Recipient email is missing."
        )

    message = EmailMessage()

    message["From"] = (
        f"CHRIST (Deemed to be University) "
        f"<{sender_email}>"
    )

    message["To"] = recipient_email

    message["Subject"] = subject

    message.set_content(
        body
    )

    # --------------------------------------------------------
    # ATTACH BOTH FILES
    # --------------------------------------------------------

    for attachment_path in (
        attachment_paths
    ):

        if not attachment_path:

            continue

        path = Path(
            attachment_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Attachment not found: {path}"
            )

        mime_type, _ = (
            mimetypes.guess_type(
                str(path)
            )
        )

        if mime_type:

            maintype, subtype = (
                mime_type.split(
                    "/",
                    1,
                )
            )

        else:

            maintype = "application"

            subtype = (
                "octet-stream"
            )

        with open(
            path,
            "rb",
        ) as attachment:

            data = (
                attachment.read()
            )

        message.add_attachment(
            data,
            maintype=maintype,
            subtype=subtype,
            filename=path.name,
        )

    # --------------------------------------------------------
    # GMAIL SMTP
    # --------------------------------------------------------

    with smtplib.SMTP(
        "smtp.gmail.com",
        587,
    ) as server:

        server.starttls()

        server.login(
            sender_email,
            sender_password,
        )

        server.send_message(
            message
        )

    return True


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        ## 🎓 AI-Powered Intelligent University Notice Distribution System
        """
    )

    st.caption(
        "Poster intelligence, RAG and personalized student communication"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📄 Upload & Process",
            "📋 Poster Metadata",
            "❓ RAG Questions",
            "👥 Students",
            "✉️ Generated Emails",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    if st.session_state.poster_name:

        st.markdown(
            "**Current Poster**"
        )

        st.caption(
            st.session_state.poster_name
        )

    if (
        st.session_state.generated_image_name
    ):

        st.markdown(
            "**Generated Invitation**"
        )

        st.caption(
            st.session_state.generated_image_name
        )

    if (
        st.session_state.students_df
        is not None
    ):

        st.markdown(
            "**Students Loaded**"
        )

        st.caption(
            f"{len(st.session_state.students_df):,}"
        )

    if st.session_state.sender_email:

        st.markdown(
            "**Sender Email**"
        )

        st.caption(
            st.session_state.sender_email
        )

    st.divider()

    if st.button(
        "🔄 Start New Notice",
        use_container_width=True,
    ):

        reset_application()

        st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="dashboard-title">
            AI-Powered Intelligent University Notice Distribution System
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-subtitle">
            AI-powered university poster processing,
            RAG and personalized student communication
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.processed:

        st.info(
            "No poster has been processed yet. "
            "Go to **Upload & Process** to begin."
        )

        columns = st.columns(4)

        with columns[0]:

            st.markdown(
                metric_card(
                    "TOTAL STUDENTS",
                    "—",
                    "Waiting for Excel",
                ),
                unsafe_allow_html=True,
            )

        with columns[1]:

            st.markdown(
                metric_card(
                    "FILTERED STUDENTS",
                    "—",
                    "Waiting for processing",
                ),
                unsafe_allow_html=True,
            )

        with columns[2]:

            st.markdown(
                metric_card(
                    "RAG QUESTIONS",
                    "—",
                    "No poster processed",
                ),
                unsafe_allow_html=True,
            )

        with columns[3]:

            st.markdown(
                metric_card(
                    "EMAILS GENERATED",
                    "—",
                    "No emails generated",
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            "### Pipeline"
        )

        st.markdown(
            """
            **1. Upload poster**

            Upload the original university poster or notice.

            **2. Upload Excel**

            Upload the student database.

            **3. Enter sender credentials**

            Enter the sender Gmail address and Gmail App Password.

            **4. Ask RAG questions**

            Enter questions that should be answered from the uploaded poster.

            **5. AI processing**

            OCR and LLM-based metadata extraction.

            **6. RAG**

            Retrieve poster information and answer the questions.

            **7. Student filtering**

            Identify students relevant to the poster.

            **8. Invitation image**

            Generate an AI-powered invitation image using Forge.

            **9. Email generation**

            Generate personalized email drafts.

            **10. Review**

            Review the generated email drafts and attachments.

            **11. Confirm and send**

            Emails are sent only after explicit confirmation.
            """
        )

    else:

        metadata = (
            st.session_state.metadata
        )

        students_df = (
            st.session_state.students_df
        )

        filtered_students = (
            st.session_state.filtered_students
        )

        rag_answers = (
            st.session_state.rag_answers
        )

        generated_emails = (
            st.session_state.generated_emails
        )

        total_students = (
            len(students_df)
            if students_df is not None
            else 0
        )

        filtered_count = (
            len(filtered_students)
            if filtered_students is not None
            else 0
        )

        not_filtered = (
            total_students
            - filtered_count
        )

        matching_rate = (
            filtered_count
            / total_students
            * 100
            if total_students
            else 0
        )

        email_count = len(
            generated_emails
        )

        st.subheader(
            safe_text(
                getattr(
                    metadata,
                    "title",
                    "Processed Poster",
                )
            )
        )

        columns = st.columns(5)

        with columns[0]:

            st.markdown(
                metric_card(
                    "TOTAL STUDENTS",
                    f"{total_students:,}",
                    "Uploaded Excel",
                ),
                unsafe_allow_html=True,
            )

        with columns[1]:

            st.markdown(
                metric_card(
                    "FILTERED STUDENTS",
                    f"{filtered_count:,}",
                    f"{matching_rate:.1f}% match",
                ),
                unsafe_allow_html=True,
            )

        with columns[2]:

            st.markdown(
                metric_card(
                    "NOT MATCHED",
                    f"{not_filtered:,}",
                    "Students not selected",
                ),
                unsafe_allow_html=True,
            )

        with columns[3]:

            st.markdown(
                metric_card(
                    "RAG QUESTIONS",
                    f"{len(rag_answers):,}",
                    "Questions answered",
                ),
                unsafe_allow_html=True,
            )

        with columns[4]:

            st.markdown(
                metric_card(
                    "EMAILS GENERATED",
                    f"{email_count:,}",
                    "Email drafts",
                ),
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown(
            "### Email Sender"
        )

        sender_email = (
            st.session_state.sender_email
        )

        if sender_email:

            st.success(
                f"Sender email: {sender_email}"
            )

        else:

            st.warning(
                "Sender email has not been configured."
            )

        left, right = st.columns(2)

        with left:

            st.markdown(
                "### Poster Information"
            )

            st.write(
                f"**Title:** "
                f"{safe_text(getattr(metadata, 'title', ''))}"
            )

            st.write(
                f"**Type:** "
                f"{safe_text(getattr(metadata, 'poster_type', ''))}"
            )

            st.write(
                f"**Department:** "
                f"{safe_text(getattr(metadata, 'department', ''))}"
            )

            st.write(
                f"**University:** "
                f"{safe_text(getattr(metadata, 'university', ''))}"
            )

            st.write(
                f"**Event Date:** "
                f"{safe_text(getattr(metadata, 'event_date', ''))}"
            )

            st.write(
                f"**Venue:** "
                f"{safe_text(getattr(metadata, 'venue', ''))}"
            )

            st.write(
                f"**Target Audience:** "
                f"{safe_text(getattr(metadata, 'target_audience', ''))}"
            )

            st.write(
                f"**Audience Type:** "
                f"{safe_text(getattr(metadata, 'target_audience_type', ''))}"
            )

        with right:

            st.markdown(
                "### AI Summary"
            )

            st.write(
                st.session_state.summary
            )

        st.divider()

        st.markdown(
            "### AI-Generated Invitation"
        )

        generated_image_path = (
            st.session_state.generated_image_path
        )

        if (
            generated_image_path
            and Path(
                generated_image_path
            ).exists()
        ):

            st.image(
                generated_image_path,
                use_container_width=True,
            )

        else:

            st.warning(
                "Generated invitation image is not available."
            )

        st.divider()

        st.markdown(
            "### RAG Questions"
        )

        if rag_answers:

            for item in rag_answers:

                with st.expander(
                    item["question"]
                ):

                    st.write(
                        item["answer"]
                    )

        else:

            st.info(
                "No RAG questions were entered."
            )

        if st.session_state.emails_sent:

            st.success(
                "Emails have been sent successfully."
            )

        else:

            st.info(
                "Email drafts have been generated. "
                "Review them in the Generated Emails section "
                "before sending."
            )


# ============================================================
# UPLOAD AND PROCESS
# ============================================================

elif page == "📄 Upload & Process":

    st.title(
        "Upload & Process"
    )

    st.write(
        "Upload the poster, student Excel file, "
        "sender credentials and RAG questions."
    )

    # ========================================================
    # SENDER EMAIL
    # ========================================================

    st.markdown(
        "## 1. Email Sender Credentials"
    )

    st.info(
        "Enter the Gmail account that should send the emails. "
        "Use a Gmail App Password, not your normal Gmail password."
    )

    sender_email_input = st.text_input(
        "Sender Gmail Address",
        value=st.session_state.sender_email,
        placeholder="example@gmail.com",
    )

    sender_password_input = st.text_input(
        "Gmail App Password",
        type="password",
        value=st.session_state.sender_password,
        placeholder="Enter your 16-character Gmail App Password",
    )

    st.caption(
        "Your App Password is masked and is not displayed on the dashboard."
    )

    # ========================================================
    # POSTER
    # ========================================================

    st.markdown(
        "## 2. Upload Poster"
    )

    poster_file = st.file_uploader(
        "Upload Poster",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
            "tiff",
            "tif",
            "pdf",
            "docx",
            "txt",
        ],
    )

    if poster_file:

        extension = Path(
            poster_file.name
        ).suffix.lower()

        st.success(
            f"Poster selected: {poster_file.name}"
        )

        if extension in IMAGE_EXTENSIONS:

            st.image(
                poster_file,
                caption="Uploaded Poster",
                use_container_width=True,
            )

        elif extension == ".pdf":

            st.info(
                "PDF poster selected."
            )

        elif extension == ".docx":

            st.info(
                "DOCX poster selected."
            )

        else:

            st.info(
                "Text poster selected."
            )

    # ========================================================
    # STUDENT EXCEL
    # ========================================================

    st.markdown(
        "## 3. Upload Student Excel"
    )

    students_file = st.file_uploader(
        "Upload Student Excel",
        type=[
            "xlsx",
            "xls",
        ],
    )

    students_df = None

    if students_file:

        try:

            students_df = pd.read_excel(
                students_file
            )

            st.success(
                f"{len(students_df):,} students loaded."
            )

            st.dataframe(
                students_df.head(10),
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                f"Excel reading failed: {error}"
            )

            st.info(
                "If openpyxl is missing, install it with: "
                "pip install openpyxl"
            )

    # ========================================================
    # RAG QUESTIONS
    # ========================================================

    st.markdown(
        "## 4. RAG Questions"
    )

    st.caption(
        "The answers to these questions will be included "
        "in the personalized email drafts."
    )

    question_count = st.number_input(
        "Number of questions",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )

    questions = []

    for index in range(
        int(question_count)
    ):

        question = st.text_input(
            f"Question {index + 1}",
            key=f"question_{index}",
            placeholder=(
                "Example: What is this event about?"
            ),
        )

        if question.strip():

            questions.append(
                question.strip()
            )

    st.divider()

    # ========================================================
    # PROCESS BUTTON
    # ========================================================

    process = st.button(
        "🚀 PROCESS POSTER",
        type="primary",
        use_container_width=True,
    )

    if process:

        # ----------------------------------------------------
        # VALIDATE EMAIL
        # ----------------------------------------------------

        if not sender_email_input.strip():

            st.error(
                "Please enter the sender Gmail address."
            )

            st.stop()

        if "@" not in sender_email_input:

            st.error(
                "Please enter a valid email address."
            )

            st.stop()

        # ----------------------------------------------------
        # VALIDATE PASSWORD
        # ----------------------------------------------------

        if not sender_password_input.strip():

            st.error(
                "Please enter the Gmail App Password."
            )

            st.stop()

        # ----------------------------------------------------
        # VALIDATE POSTER
        # ----------------------------------------------------

        if poster_file is None:

            st.error(
                "Please upload a poster."
            )

            st.stop()

        # ----------------------------------------------------
        # VALIDATE STUDENTS
        # ----------------------------------------------------

        if students_df is None:

            st.error(
                "Please upload the student Excel file."
            )

            st.stop()

        if students_df.empty:

            st.error(
                "The Excel file contains no students."
            )

            st.stop()

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = {
            "student_id",
            "name",
            "email",
            "programme",
            "department",
            "level",
            "interests",
        }

        missing_columns = (
            required_columns
            - set(
                students_df.columns
            )
        )

        if missing_columns:

            st.error(
                "Missing required student columns: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

            st.stop()

        try:

            with st.status(
                "Running AI pipeline...",
                expanded=True,
            ) as status:

                # ====================================================
                # SAVE CREDENTIALS
                # ====================================================

                st.write(
                    "Saving sender configuration..."
                )

                st.session_state.sender_email = (
                    sender_email_input.strip()
                )

                st.session_state.sender_password = (
                    sender_password_input.strip()
                )

                st.session_state.emails_sent = False

                st.session_state.email_send_results = []

                # ====================================================
                # SAVE POSTER
                # ====================================================

                st.write(
                    "Saving original poster..."
                )

                poster_path = (
                    save_uploaded_file(
                        poster_file
                    )
                )

                st.session_state.poster_path = (
                    poster_path
                )

                st.session_state.poster_name = (
                    poster_file.name
                )

                # ====================================================
                # INITIALIZE COMPONENTS
                # ====================================================

                st.write(
                    "Initializing AI components..."
                )

                loader = (
                    get_document_loader()
                )

                # IMPORTANT:
                # Do not initialize Gemma vision model.

                vision_model = (
                    get_vision_model()
                )

                extractor = (
                    get_metadata_extractor()
                )

                rag = (
                    get_rag()
                )

                summarizer = (
                    get_summarizer()
                )

                email_generator = (
                    get_email_generator()
                )

                # ====================================================
                # OCR
                # ====================================================

                st.write(
                    "Extracting poster text using OCR..."
                )

                ocr_text = (
                    extract_ocr_text(
                        loader,
                        poster_path,
                    )
                )

                if not ocr_text.strip():

                    st.warning(
                        "OCR did not extract any text. "
                        "Metadata extraction may have limited information."
                    )

                # ====================================================
                # VISION
                # ====================================================

                st.write(
                    "Skipping Gemma vision analysis due to memory constraints..."
                )

                vlm_output = (
                    run_vision_analysis(
                        vision_model,
                        poster_path,
                    )
                )

                # ====================================================
                # METADATA
                # ====================================================

                st.write(
                    "Extracting structured metadata..."
                )

                raw_metadata = (
                    extractor.extract_metadata(
                        ocr_text=ocr_text,
                        vlm_output=vlm_output,
                    )
                )

                # ====================================================
                # NORMALIZATION
                # ====================================================

                st.write(
                    "Normalizing extracted metadata..."
                )

                cleaned_metadata = (
                    normalize_metadata_for_pydantic(
                        raw_metadata
                    )
                )

                # ====================================================
                # PYDANTIC
                # ====================================================

                st.write(
                    "Validating extracted metadata..."
                )

                try:

                    metadata = PosterMetadata(
                        **cleaned_metadata
                    )

                except Exception as metadata_error:

                    st.error(
                        "Metadata validation failed."
                    )

                    st.json(
                        cleaned_metadata
                    )

                    raise metadata_error

                st.session_state.metadata = (
                    metadata
                )

                # ====================================================
                # RAG CONTEXT
                # ====================================================

                st.write(
                    "Retrieving poster information using RAG..."
                )

                rag_context = (
                    rag.retrieve_context(
                        "Retrieve all relevant information "
                        "from the current poster needed for "
                        "student communication, including "
                        "event details, purpose, audience, "
                        "eligibility, important dates, venue, "
                        "registration information, publication "
                        "opportunities and other useful facts."
                    )
                )

                # ====================================================
                # RAG QUESTIONS
                # ====================================================

                rag_answers = []

                if questions:

                    st.write(
                        f"Answering {len(questions)} RAG question(s)..."
                    )

                    for question in questions:

                        answer = (
                            rag.ask(
                                question
                            )
                        )

                        rag_answers.append(
                            {
                                "question": question,
                                "answer": answer,
                            }
                        )

                st.session_state.rag_answers = (
                    rag_answers
                )

                question_context = (
                    create_rag_context(
                        rag_answers
                    )
                )

                if question_context:

                    final_rag_context = (
                        str(
                            rag_context
                        )
                        + "\n\n"
                        + question_context
                    )

                else:

                    final_rag_context = (
                        str(
                            rag_context
                        )
                    )

                st.session_state.rag_context = (
                    final_rag_context
                )

                # ====================================================
                # SUMMARY
                # ====================================================

                st.write(
                    "Generating poster summary..."
                )

                summary = (
                    summarizer.generate_summary(
                        metadata
                    )
                )

                st.session_state.summary = (
                    summary
                )

                # ====================================================
                # FORGE IMAGE GENERATION
                # ====================================================

                st.write(
                    "Generating AI invitation image using Forge..."
                )

                poster_title = safe_text(
                    getattr(
                        metadata,
                        "title",
                        "University Event",
                    )
                )

                poster_description = safe_text(
                    getattr(
                        metadata,
                        "description",
                        "",
                    )
                )

                poster_type = safe_text(
                    getattr(
                        metadata,
                        "poster_type",
                        "academic event",
                    )
                )

                event_date = safe_text(
                    getattr(
                        metadata,
                        "event_date",
                        "",
                    )
                )

                venue = safe_text(
                    getattr(
                        metadata,
                        "venue",
                        "",
                    )
                )

                image_prompt = (
                    "Create a professional modern "
                    "university event invitation background "
                    f"for an academic {poster_type}. "

                    f"Event: {poster_title}. "

                    f"Description: {poster_description}. "

                    f"Date: {event_date}. "

                    f"Venue: {venue}. "

                    "Elegant university atmosphere, "
                    "professional academic environment, "
                    "students where appropriate, "
                    "cinematic lighting, "
                    "professional institutional design, "
                    "high quality, clean composition, "
                    "visually attractive, "
                    "suitable as an invitation background, "
                    "leave clear visual space for text, "
                    "no text, no letters, no words, "
                    "no watermark."
                )

                image_negative_prompt = (
                    "text, letters, words, watermark, "
                    "blurry, low quality, distorted, "
                    "deformed, duplicate people, "
                    "bad anatomy, ugly, noisy"
                )

                generated_image_path = (
                    generate_invitation_image(
                        prompt=image_prompt,
                        negative_prompt=(
                            image_negative_prompt
                        ),
                    )
                )

                st.session_state.generated_image_path = (
                    generated_image_path
                )

                st.session_state.generated_image_name = (
                    Path(
                        generated_image_path
                    ).name
                )

                st.write(
                    "AI invitation image generated successfully."
                )

                # ====================================================
                # STUDENT FILTERING
                # ====================================================

                st.write(
                    "Filtering relevant students..."
                )

                student_filter = StudentFilter(
                    students_df
                )

                filtered_students = (
                    student_filter.filter_students(
                        metadata
                    )
                )

                st.session_state.students_df = (
                    students_df
                )

                st.session_state.filtered_students = (
                    filtered_students
                )

                # ====================================================
                # EMAIL DRAFT GENERATION
                # ====================================================

                st.write(
                    "Generating personalized email drafts..."
                )

                generated_emails = []

                for _, student in (
                    filtered_students.iterrows()
                ):

                    email_result = (
                        call_email_generator(
                            generator=email_generator,
                            metadata=metadata,
                            summary=summary,
                            student=student,
                            rag_context=final_rag_context,
                        )
                    )

                    email = (
                        normalize_email_result(
                            email_result
                        )
                    )

                    generated_emails.append(
                        {
                            "student_id": safe_text(
                                student.get(
                                    "student_id",
                                    "",
                                )
                            ),

                            "name": safe_text(
                                student.get(
                                    "name",
                                    "",
                                )
                            ),

                            "email": safe_text(
                                student.get(
                                    "email",
                                    "",
                                )
                            ),

                            "subject": email[
                                "subject"
                            ],

                            "body": email[
                                "body"
                            ],

                            # Original poster
                            "attachment_path": (
                                poster_path
                            ),

                            "attachment_name": (
                                poster_file.name
                            ),

                            # Generated invitation
                            "generated_image_path": (
                                generated_image_path
                            ),

                            "generated_image_name": (
                                Path(
                                    generated_image_path
                                ).name
                            ),
                        }
                    )

                st.session_state.generated_emails = (
                    generated_emails
                )

                # ====================================================
                # DO NOT SEND HERE
                # ====================================================

                st.write(
                    "Email drafts generated successfully."
                )

                st.write(
                    "Original poster and AI-generated invitation "
                    "are ready as email attachments."
                )

                st.write(
                    "Waiting for user confirmation before sending emails."
                )

                # ====================================================
                # COMPLETE
                # ====================================================

                st.session_state.processed = True

                status.update(
                    label=(
                        "Pipeline completed successfully"
                    ),
                    state="complete",
                )

            st.success(
                "Poster processing completed successfully."
            )

            st.info(
                "Email drafts are ready. "
                "Go to **Generated Emails** to review them "
                "and confirm sending."
            )

        except Exception as error:

            st.error(
                "Pipeline encountered an error."
            )

            st.error(
                f"{type(error).__name__}: {error}"
            )

            st.exception(
                error
            )


# ============================================================
# POSTER METADATA
# ============================================================

elif page == "📋 Poster Metadata":

    st.title(
        "Poster Metadata"
    )

    metadata = (
        st.session_state.metadata
    )

    if metadata is None:

        st.info(
            "Process a poster first."
        )

    else:

        st.dataframe(
            metadata_dataframe(
                metadata
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "### Original Poster"
        )

        poster_path = (
            st.session_state.poster_path
        )

        if poster_path:

            extension = Path(
                poster_path
            ).suffix.lower()

            if extension in IMAGE_EXTENSIONS:

                st.image(
                    poster_path,
                    use_container_width=True,
                )

            else:

                st.write(
                    f"**File:** "
                    f"{st.session_state.poster_name}"
                )

        st.markdown(
            "### AI-Generated Invitation"
        )

        generated_image_path = (
            st.session_state.generated_image_path
        )

        if (
            generated_image_path
            and Path(
                generated_image_path
            ).exists()
        ):

            st.image(
                generated_image_path,
                use_container_width=True,
            )

        else:

            st.info(
                "No generated invitation image available."
            )


# ============================================================
# RAG QUESTIONS
# ============================================================

elif page == "❓ RAG Questions":

    st.title(
        "RAG Questions & Answers"
    )

    if not st.session_state.processed:

        st.info(
            "Process a poster first."
        )

    else:

        answers = (
            st.session_state.rag_answers
        )

        if not answers:

            st.info(
                "No questions were entered."
            )

        else:

            for index, item in enumerate(
                answers,
                start=1,
            ):

                st.markdown(
                    f"### Question {index}"
                )

                st.write(
                    item["question"]
                )

                st.markdown(
                    "**RAG Answer**"
                )

                st.write(
                    item["answer"]
                )

                st.divider()


# ============================================================
# STUDENTS
# ============================================================

elif page == "👥 Students":

    st.title(
        "Student Matching"
    )

    students_df = (
        st.session_state.students_df
    )

    filtered_students = (
        st.session_state.filtered_students
    )

    if (
        students_df is None
        or filtered_students is None
    ):

        st.info(
            "Process a poster first."
        )

    else:

        total = len(
            students_df
        )

        filtered = len(
            filtered_students
        )

        not_filtered = (
            total - filtered
        )

        matching_rate = (
            filtered
            / total
            * 100
            if total
            else 0
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Total Students",
            total,
        )

        c2.metric(
            "Filtered Students",
            filtered,
        )

        c3.metric(
            "Not Matched",
            not_filtered,
        )

        c4.metric(
            "Matching Rate",
            f"{matching_rate:.1f}%",
        )

        st.divider()

        st.markdown(
            "### Filtered Students"
        )

        st.dataframe(
            filtered_students,
            use_container_width=True,
            hide_index=True,
        )

        csv = (
            filtered_students.to_csv(
                index=False
            )
        )

        st.download_button(
            "⬇️ Download Filtered Students",
            data=csv,
            file_name="filtered_students.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# GENERATED EMAILS
# ============================================================

elif page == "✉️ Generated Emails":

    st.title(
        "Generated Emails"
    )

    emails = (
        st.session_state.generated_emails
    )

    sender_email = (
        st.session_state.sender_email
    )

    if not st.session_state.processed:

        st.info(
            "Process a poster first."
        )

    elif not emails:

        st.warning(
            "No emails were generated because no students matched."
        )

    else:

        st.success(
            f"{len(emails):,} personalized email drafts generated."
        )

        if sender_email:

            st.markdown(
                f"### Sender\n\n"
                f"**{sender_email}**"
            )

        st.markdown(
            f"### Recipients\n\n"
            f"**{len(emails):,} students**"
        )

        st.divider()

        # ====================================================
        # EMAIL DRAFTS
        # ====================================================

        st.markdown(
            "## Review Email Drafts"
        )

        st.info(
            "Review the generated emails below. "
            "Each email will contain both the original "
            "poster and the AI-generated invitation. "
            "Emails will NOT be sent until you explicitly "
            "confirm the sending operation."
        )

        for index, email in enumerate(
            emails,
            start=1,
        ):

            with st.expander(
                f'{index}. {email["name"]} — {email["email"]}'
            ):

                st.markdown(
                    f"**Recipient:** "
                    f"{email['email']}"
                )

                st.markdown(
                    f"**Subject:** "
                    f"{email['subject']}"
                )

                st.text_area(
                    "Email Body",
                    value=email["body"],
                    height=450,
                    key=f"email_body_{index}",
                )

                # ====================================================
                # ORIGINAL POSTER
                # ====================================================

                attachment_path = (
                    email.get(
                        "attachment_path"
                    )
                )

                attachment_name = (
                    email.get(
                        "attachment_name"
                    )
                )

                st.markdown(
                    "### 📎 Original Poster"
                )

                if (
                    attachment_path
                    and Path(
                        attachment_path
                    ).exists()
                ):

                    st.write(
                        f"**Attachment:** "
                        f"{attachment_name}"
                    )

                    extension = (
                        Path(
                            attachment_path
                        ).suffix.lower()
                    )

                    if extension in (
                        IMAGE_EXTENSIONS
                    ):

                        st.image(
                            attachment_path,
                            use_container_width=True,
                        )

                    with open(
                        attachment_path,
                        "rb",
                    ) as attachment_file:

                        attachment_data = (
                            attachment_file.read()
                        )

                    st.download_button(
                        "⬇️ Download Original Poster",
                        data=attachment_data,
                        file_name=attachment_name,
                        key=f"attachment_{index}",
                    )

                else:

                    st.warning(
                        "Original poster attachment could not be found."
                    )

                # ====================================================
                # GENERATED INVITATION
                # ====================================================

                generated_image_path = (
                    email.get(
                        "generated_image_path"
                    )
                )

                generated_image_name = (
                    email.get(
                        "generated_image_name"
                    )
                )

                st.markdown(
                    "### 🎨 AI-Generated Invitation"
                )

                if (
                    generated_image_path
                    and Path(
                        generated_image_path
                    ).exists()
                ):

                    st.image(
                        generated_image_path,
                        caption=generated_image_name,
                        use_container_width=True,
                    )

                    with open(
                        generated_image_path,
                        "rb",
                    ) as generated_file:

                        generated_data = (
                            generated_file.read()
                        )

                    st.download_button(
                        "⬇️ Download Generated Invitation",
                        data=generated_data,
                        file_name=generated_image_name,
                        key=(
                            f"generated_attachment_{index}"
                        ),
                    )

                else:

                    st.warning(
                        "Generated invitation image could not be found."
                    )

        # ====================================================
        # SEND SECTION
        # ====================================================

        st.divider()

        st.markdown(
            "## Send Emails"
        )

        if st.session_state.emails_sent:

            st.success(
                "Emails have already been sent for this notice."
            )

            results = (
                st.session_state.email_send_results
            )

            if results:

                st.dataframe(
                    pd.DataFrame(
                        results
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            st.warning(
                "This action will send real emails to the "
                "students shown above."
            )

            st.markdown(
                f"""
                **Sender:** {sender_email}

                **Recipients:** {len(emails):,}

                **Attachments:**
                - Original uploaded poster
                - AI-generated invitation image
                """
            )

            confirm = st.checkbox(
                "I have reviewed the generated email drafts and I confirm that these emails should be sent.",
                key="confirm_email_sending",
            )

            if confirm:

                st.success(
                    "Confirmation received. "
                    "You can now send the emails."
                )

            send_button = st.button(
                "📨 CONFIRM & SEND EMAILS",
                type="primary",
                use_container_width=True,
                disabled=not confirm,
            )

            if send_button:

                # ====================================================
                # FINAL VALIDATION
                # ====================================================

                if not sender_email:

                    st.error(
                        "Sender email is missing."
                    )

                    st.stop()

                if not st.session_state.sender_password:

                    st.error(
                        "Sender App Password is missing."
                    )

                    st.stop()

                if not confirm:

                    st.error(
                        "Please confirm before sending."
                    )

                    st.stop()

                # ====================================================
                # SEND EMAILS
                # ====================================================

                try:

                    with st.status(
                        "Sending emails...",
                        expanded=True,
                    ) as send_status:

                        st.write(
                            "Connecting to Gmail SMTP..."
                        )

                        email_results = []

                        for index, email_data in enumerate(
                            emails,
                            start=1,
                        ):

                            st.write(
                                f"Sending email {index} "
                                f"of {len(emails)} "
                                f"to {email_data['email']}..."
                            )

                            try:

                                # ------------------------------------------------
                                # BODY FROM UI
                                # ------------------------------------------------

                                body = st.session_state.get(
                                    f"email_body_{index}",
                                    email_data["body"],
                                )

                                # ------------------------------------------------
                                # TWO ATTACHMENTS
                                # ------------------------------------------------

                                attachment_paths = [

                                    email_data.get(
                                        "attachment_path"
                                    ),

                                    email_data.get(
                                        "generated_image_path"
                                    ),

                                ]

                                send_email_with_attachments(
                                    sender_email=(
                                        sender_email
                                    ),

                                    sender_password=(
                                        st.session_state
                                        .sender_password
                                    ),

                                    recipient_email=(
                                        email_data[
                                            "email"
                                        ]
                                    ),

                                    recipient_name=(
                                        email_data[
                                            "name"
                                        ]
                                    ),

                                    subject=(
                                        email_data[
                                            "subject"
                                        ]
                                    ),

                                    body=body,

                                    attachment_paths=(
                                        attachment_paths
                                    ),
                                )

                                email_results.append(
                                    {
                                        "student_id": (
                                            email_data[
                                                "student_id"
                                            ]
                                        ),

                                        "name": (
                                            email_data[
                                                "name"
                                            ]
                                        ),

                                        "email": (
                                            email_data[
                                                "email"
                                            ]
                                        ),

                                        "status": "Sent",

                                        "attachments": (
                                            "Original Poster + "
                                            "AI Invitation"
                                        ),

                                        "error": "",
                                    }
                                )

                            except Exception as send_error:

                                email_results.append(
                                    {
                                        "student_id": (
                                            email_data[
                                                "student_id"
                                            ]
                                        ),

                                        "name": (
                                            email_data[
                                                "name"
                                            ]
                                        ),

                                        "email": (
                                            email_data[
                                                "email"
                                            ]
                                        ),

                                        "status": "Failed",

                                        "attachments": (
                                            "Original Poster + "
                                            "AI Invitation"
                                        ),

                                        "error": str(
                                            send_error
                                        ),
                                    }
                                )

                        # ====================================================
                        # RESULTS
                        # ====================================================

                        st.session_state.email_send_results = (
                            email_results
                        )

                        successful = sum(
                            1
                            for result in email_results
                            if result["status"]
                            == "Sent"
                        )

                        failed = (
                            len(email_results)
                            - successful
                        )

                        if failed == 0:

                            st.session_state.emails_sent = (
                                True
                            )

                            st.session_state.email_send_confirmed = (
                                True
                            )

                            send_status.update(
                                label=(
                                    "All emails sent successfully"
                                ),
                                state="complete",
                            )

                        else:

                            send_status.update(
                                label=(
                                    "Email sending completed with some failures"
                                ),
                                state="error",
                            )

                    # ====================================================
                    # DISPLAY RESULTS
                    # ====================================================

                    if failed == 0:

                        st.success(
                            f"Successfully sent "
                            f"{successful:,} email(s)."
                        )

                    else:

                        st.warning(
                            f"Sent {successful:,} email(s), "
                            f"but {failed:,} email(s) failed."
                        )

                    st.dataframe(
                        pd.DataFrame(
                            email_results
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                except Exception as send_error:

                    st.error(
                        "Email sending failed."
                    )

                    st.error(
                        f"{type(send_error).__name__}: "
                        f"{send_error}"
                    )

                    st.exception(
                        send_error
                    )

        st.divider()

        st.info(
            "Each outgoing email contains the original "
            "uploaded poster and the AI-generated invitation "
            "image. Emails are sent only after explicit "
            "user confirmation."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        AI-Powered Intelligent University Notice Distribution System
        • OCR + LLM + RAG + Student Matching
        + Personalized Email Generation
        + Forge AI Invitation Generation
        + Gmail Email Delivery
    </div>
    """,
    unsafe_allow_html=True,
)