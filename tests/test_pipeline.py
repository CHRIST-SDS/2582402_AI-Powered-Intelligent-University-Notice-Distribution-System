import os
import sys
import inspect
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT COMPONENTS
# ============================================================

from src.extraction.schemas import PosterMetadata
from src.rag.rag_pipeline import PosterRAG
from src.summarization.summarizer import PosterSummarizer
from src.student_filtering.student_filter import StudentFilter
from src.email.email_generator import EmailGenerator
from src.email.email_sender import EmailSender


# ============================================================
# TEST POSTER PATH
# ============================================================

POSTER_PATH = (
    r"C:\Python_project\Generative AI"
    r"\Christ_University_Posters"
    r"\christ_poster_1.jpg"
)


# ============================================================
# CREATE TEST METADATA
# ============================================================

def create_test_metadata():

    return PosterMetadata(

        poster_type="Conference",

        title="FUTURE-ED GLOBAL CONFERENCE",

        department="School of Education",

        university="CHRIST (Deemed to be University)",

        description=(
            "The poster describes an international conference "
            "focused on international perspectives in shaping "
            "learning for future generations."
        ),

        short_summary=(
            "An international conference focused on "
            "shaping learning for future generations."
        ),

        event_date="28 & 29 August 2026",

        important_dates=[
            "28 & 29 August 2026"
        ],

        venue=(
            "CHRIST (Deemed to be University) "
            "Bangalore Central Campus"
        ),

        registration_deadline="Not Available",

        registration_link="Not Available",

        guest_speakers=[],

        contact_person="Not Available",

        contact_phone=[],

        email="Not Available",

        eligibility="Not Available",

        target_audience="Researchers and academicians",

        target_audience_type="Inferred",

        keywords=[
            "Education",
            "Future-Ed",
            "Learning",
            "Conference"
        ],

        other_information=(
            "Publication Opportunities, "
            "Scopus indexed Emerald Journals"
        )
    )


# ============================================================
# CREATE TEST STUDENT DATA
# ============================================================

def create_test_students():

    return pd.DataFrame([

        {
            "student_id": "ST001",
            "name": "Rahul Sharma",
            "email": "rahul@email.com",
            "programme": "MSc Education",
            "department": "School of Education",
            "level": "postgraduate",
            "interests": (
                "Education, Learning, "
                "Research, Pedagogy"
            )
        },

        {
            "student_id": "ST002",
            "name": "Ananya Rao",
            "email": "ananya@email.com",
            "programme": "BSc Psychology",
            "department": "Psychology",
            "level": "undergraduate",
            "interests": (
                "Education, Learning"
            )
        },

        {
            "student_id": "ST003",
            "name": "Arjun Kumar",
            "email": "arjun@email.com",
            "programme": "BCom",
            "department": "Commerce",
            "level": "undergraduate",
            "interests": (
                "Finance, Accounting"
            )
        },

        {
            "student_id": "ST004",
            "name": "Priya Nair",
            "email": "priya@email.com",
            "programme": "PhD Education",
            "department": "School of Education",
            "level": "doctoral",
            "interests": (
                "Education, Research"
            )
        }
    ])


# ============================================================
# TEST 1
# METADATA VALIDATION
# ============================================================

def test_metadata():

    print()
    print("=" * 70)
    print("TEST 1: METADATA VALIDATION")
    print("=" * 70)

    metadata = create_test_metadata()

    assert isinstance(
        metadata,
        PosterMetadata
    )

    assert (
        metadata.title
        == "FUTURE-ED GLOBAL CONFERENCE"
    )

    assert (
        metadata.target_audience
        == "Researchers and academicians"
    )

    assert (
        metadata.target_audience_type
        == "Inferred"
    )

    print()
    print(
        "PosterMetadata validation successful."
    )

    print()
    print(
        "Target Audience:",
        metadata.target_audience
    )

    print(
        "Target Audience Type:",
        metadata.target_audience_type
    )

    print()
    print("TEST 1 PASSED")

    return metadata


# ============================================================
# TEST 2
# SINGLE-POSTER RAG RETRIEVAL
# ============================================================

def test_rag():

    print()
    print("=" * 70)
    print("TEST 2: SINGLE-POSTER RAG RETRIEVAL")
    print("=" * 70)

    rag = PosterRAG()

    question = (
        "Retrieve the most relevant information needed "
        "to understand this poster, including its "
        "purpose, event details, audience, dates, venue, "
        "publication opportunities and other important "
        "information."
    )

    rag_context = rag.retrieve_context(
        question
    )

    print()
    print(
        "RAG retrieval completed."
    )

    print()
    print(
        "========== RAG CONTEXT =========="
    )

    print(
        rag_context
    )

    assert rag_context is not None

    assert len(
        str(rag_context).strip()
    ) > 0

    print()
    print("TEST 2 PASSED")

    return rag_context


# ============================================================
# TEST 3
# SUMMARY GENERATION
# ============================================================

def test_summary(
    metadata
):

    print()
    print("=" * 70)
    print("TEST 3: SUMMARY GENERATION")
    print("=" * 70)

    summarizer = PosterSummarizer()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The actual method in your project is:
    #
    # generate_summary(metadata)
    #
    # RAG retrieval is handled internally by the
    # PosterSummarizer.
    # --------------------------------------------------------

    summary = summarizer.generate_summary(
        metadata
    )

    print()
    print(
        "========== GENERATED SUMMARY =========="
    )

    print(
        summary
    )

    assert summary is not None

    assert isinstance(
        summary,
        str
    )

    assert len(
        summary.strip()
    ) > 0

    print()
    print(
        "Summary successfully generated."
    )

    print()
    print("TEST 3 PASSED")

    return summary


# ============================================================
# TEST 4
# STUDENT FILTERING
# ============================================================

def test_student_filter(
    metadata
):

    print()
    print("=" * 70)
    print("TEST 4: STUDENT FILTERING")
    print("=" * 70)

    students = create_test_students()

    print()
    print(
        "Total students:"
    )

    print(
        len(students)
    )

    # --------------------------------------------------------
    # ACTUAL StudentFilter CONSTRUCTOR
    #
    # StudentFilter(students_df)
    # --------------------------------------------------------

    student_filter = StudentFilter(
        students
    )

    # --------------------------------------------------------
    # FILTER STUDENTS
    # --------------------------------------------------------

    filtered_students = (
        student_filter.filter_students(
            metadata
        )
    )

    print()
    print(
        "Matching students:"
    )

    print(
        len(filtered_students)
    )

    print()
    print(
        "========== FILTERED STUDENTS =========="
    )

    print(
        filtered_students.to_string(
            index=False
        )
    )

    assert filtered_students is not None

    assert isinstance(
        filtered_students,
        pd.DataFrame
    )

    assert len(
        filtered_students
    ) > 0

    student_ids = (
        filtered_students[
            "student_id"
        ].tolist()
    )

    assert "ST001" in student_ids

    assert "ST004" in student_ids

    print()
    print(
        "Rahul Sharma correctly identified as relevant."
    )

    print(
        "Priya Nair correctly identified as relevant."
    )

    print()
    print(
        "TEST 4 PASSED"
    )

    return filtered_students


# ============================================================
# FIND EMAIL GENERATION METHOD
# ============================================================

def get_email_generation_method():

    generator = EmailGenerator()

    # --------------------------------------------------------
    # FIRST CHECK THE EXPECTED METHOD
    # --------------------------------------------------------

    if hasattr(
        generator,
        "generate_email"
    ):

        return (
            generator,
            generator.generate_email
        )

    # --------------------------------------------------------
    # OTHERWISE SEARCH FOR PUBLIC EMAIL METHOD
    # --------------------------------------------------------

    candidates = []

    for name, method in inspect.getmembers(
        generator,
        predicate=callable
    ):

        if name.startswith("_"):
            continue

        if "email" in name.lower():

            candidates.append(
                (
                    name,
                    method
                )
            )

    if len(candidates) == 1:

        name, method = candidates[0]

        print()
        print(
            "Detected email generation method:",
            name
        )

        return (
            generator,
            method
        )

    available_methods = [

        name

        for name, method
        in inspect.getmembers(
            generator,
            predicate=callable
        )

        if not name.startswith("_")
    ]

    raise AttributeError(
        "\nCould not identify the email "
        "generation method.\n\n"
        f"Available methods: {available_methods}"
    )


# ============================================================
# GENERATE ONE EMAIL
# ============================================================

def generate_one_email(
    method,
    metadata,
    summary,
    rag_context,
    student
):

    signature = inspect.signature(
        method
    )

    parameters = signature.parameters

    available_values = {

        "metadata": metadata,

        "poster_metadata": metadata,

        "summary": summary,

        "rag_context": rag_context,

        "retrieved_context": rag_context,

        "context": rag_context,

        "student": student,

        "student_data": student,

        "student_profile": student
    }

    kwargs = {}

    missing = []

    for name, parameter in parameters.items():

        if name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD
        ):
            continue

        if name in available_values:

            kwargs[name] = (
                available_values[name]
            )

        elif (
            parameter.default
            is inspect.Parameter.empty
        ):

            missing.append(
                name
            )

    if missing:

        raise TypeError(
            "\nThe actual EmailGenerator method "
            "requires parameters that were not "
            "identified by this test.\n\n"
            f"Missing parameters: {missing}\n\n"
            f"Actual signature: {signature}"
        )

    print()
    print(
        "Email generation method signature:"
    )

    print(
        signature
    )

    return method(
        **kwargs
    )


# ============================================================
# TEST 5
# PERSONALIZED EMAIL GENERATION
# ============================================================

def test_email_generation(
    metadata,
    summary,
    rag_context,
    filtered_students
):

    print()
    print("=" * 70)
    print(
        "TEST 5: PERSONALIZED EMAIL GENERATION"
    )
    print("=" * 70)

    generator, method = (
        get_email_generation_method()
    )

    generated_emails = []

    # --------------------------------------------------------
    # GENERATE EMAIL FOR EACH FILTERED STUDENT
    # --------------------------------------------------------

    for _, student in (
        filtered_students.iterrows()
    ):

        student_data = (
            student.to_dict()
        )

        print()
        print(
            "-" * 70
        )

        print(
            "Generating email for:",
            student_data["name"]
        )

        email_body = generate_one_email(

            method=method,

            metadata=metadata,

            summary=summary,

            rag_context=rag_context,

            student=student_data
        )

        # ----------------------------------------------------
        # ACTUAL EMAIL GENERATOR OUTPUT
        #
        # It returns the email body string.
        # ----------------------------------------------------

        assert email_body is not None

        assert isinstance(
            email_body,
            str
        )

        assert len(
            email_body.strip()
        ) > 0

        # ----------------------------------------------------
        # PERSONALIZATION
        # ----------------------------------------------------

        assert (
            str(student_data["name"])
            in email_body
        )

        generated_emails.append(
            email_body
        )

        print()
        print(
            "========== GENERATED EMAIL =========="
        )

        print(
            email_body
        )

    # --------------------------------------------------------
    # EMAIL COUNT
    # --------------------------------------------------------

    assert (
        len(generated_emails)
        == len(filtered_students)
    )

    print()
    print(
        "Generated "
        f"{len(generated_emails)} "
        "personalized emails."
    )

    print()
    print("TEST 5 PASSED")

    return generated_emails


# ============================================================
# TEST 6
# EMAIL + POSTER ATTACHMENT
# ============================================================

def test_email_attachment(
    generated_emails,
    filtered_students
):

    print()
    print("=" * 70)
    print(
        "TEST 6: EMAIL + POSTER ATTACHMENT"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # VERIFY POSTER EXISTS
    # --------------------------------------------------------

    assert os.path.exists(
        POSTER_PATH
    ), (
        f"Poster not found: {POSTER_PATH}"
    )

    print()
    print(
        "Poster verified:"
    )

    print(
        POSTER_PATH
    )

    # --------------------------------------------------------
    # DETERMINE ACTUAL POSTER TYPE
    # --------------------------------------------------------

    poster_extension = (
        os.path.splitext(
            POSTER_PATH
        )[1]
        .lower()
    )

    print()
    print(
        "Poster file extension:",
        poster_extension
    )

    # --------------------------------------------------------
    # SUPPORTED MIME TYPES
    # --------------------------------------------------------

    expected_content_types = {

        ".pdf": "application/pdf",

        ".jpg": "image/jpeg",

        ".jpeg": "image/jpeg",

        ".png": "image/png"
    }

    assert (
        poster_extension
        in expected_content_types
    ), (
        "Unsupported poster file type: "
        f"{poster_extension}"
    )

    expected_content_type = (
        expected_content_types[
            poster_extension
        ]
    )

    print(
        "Expected MIME type:",
        expected_content_type
    )

    # --------------------------------------------------------
    # CREATE TEST EMAIL SENDER
    # --------------------------------------------------------

    sender = EmailSender(

        smtp_server="smtp.test.com",

        smtp_port=587,

        sender_email="sender@test.com",

        sender_password="test_password"
    )

    # --------------------------------------------------------
    # EMAIL COUNT MUST MATCH FILTERED STUDENTS
    # --------------------------------------------------------

    assert (
        len(generated_emails)
        == len(filtered_students)
    )

    # --------------------------------------------------------
    # TEST EACH GENERATED EMAIL
    # --------------------------------------------------------

    for index, (
        email_body,
        (_, student)
    ) in enumerate(

        zip(
            generated_emails,
            filtered_students.iterrows()
        ),

        start=1
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"EMAIL {index}"
        )

        print(
            "Student:",
            student["name"]
        )

        print(
            "Recipient:",
            student["email"]
        )

        # ----------------------------------------------------
        # VERIFY BODY
        # ----------------------------------------------------

        assert email_body is not None

        assert isinstance(
            email_body,
            str
        )

        assert len(
            email_body.strip()
        ) > 0

        # ----------------------------------------------------
        # VERIFY PERSONALIZATION
        # ----------------------------------------------------

        assert (
            str(student["name"])
            in email_body
        )

        print(
            "Personalization verified."
        )

        # ----------------------------------------------------
        # CREATE MIME MESSAGE
        #
        # This creates the message only.
        # It DOES NOT send a real email.
        # ----------------------------------------------------

        message = sender._create_message(

            recipient_email=student["email"],

            recipient_name=student["name"],

            subject=(
                "Academic Opportunity: "
                "FUTURE-ED GLOBAL CONFERENCE"
            ),

            body=email_body,

            poster_path=POSTER_PATH
        )

        # ----------------------------------------------------
        # VERIFY RECIPIENT
        # ----------------------------------------------------

        assert (
            student["email"]
            in message["To"]
        )

        print(
            "Recipient verified."
        )

        # ----------------------------------------------------
        # VERIFY SUBJECT
        # ----------------------------------------------------

        assert (
            message["Subject"]
            == (
                "Academic Opportunity: "
                "FUTURE-ED GLOBAL CONFERENCE"
            )
        )

        print(
            "Subject verified."
        )

        # ----------------------------------------------------
        # FIND ATTACHMENTS
        # ----------------------------------------------------

        attachments = [

            part

            for part in message.walk()

            if (
                part.get_content_disposition()
                == "attachment"
            )
        ]

        print()
        print(
            "Number of attachments:",
            len(attachments)
        )

        # ----------------------------------------------------
        # EXACTLY ONE ATTACHMENT
        # ----------------------------------------------------

        assert (
            len(attachments)
            == 1
        )

        attachment = attachments[0]

        # ----------------------------------------------------
        # VERIFY FILENAME
        # ----------------------------------------------------

        attachment_filename = (
            attachment.get_filename()
        )

        print(
            "Attachment filename:",
            attachment_filename
        )

        assert (
            attachment_filename
            == os.path.basename(
                POSTER_PATH
            )
        )

        print(
            "Attachment filename verified."
        )

        # ----------------------------------------------------
        # VERIFY CONTENT TYPE
        # ----------------------------------------------------

        actual_content_type = (
            attachment.get_content_type()
        )

        print(
            "Attachment content type:",
            actual_content_type
        )

        assert (
            actual_content_type
            == expected_content_type
        )

        print(
            "Attachment MIME type verified."
        )

        # ----------------------------------------------------
        # VERIFY ATTACHMENT CONTENT
        # ----------------------------------------------------

        attachment_payload = (
            attachment.get_payload(
                decode=True
            )
        )

        assert (
            attachment_payload
            is not None
        )

        assert (
            len(attachment_payload)
            > 0
        )

        print(
            "Attachment content verified."
        )

        print(
            "Poster successfully attached."
        )

    # --------------------------------------------------------
    # TEST COMPLETE
    # --------------------------------------------------------

    print()
    print(
        "Original uploaded poster successfully "
        "attached to every generated email."
    )

    print()
    print("TEST 6 PASSED")


# ============================================================
# FULL PIPELINE TEST
# ============================================================

def run_pipeline_test():

    print()
    print(
        "#" * 70
    )

    print(
        "POSTER SUMMARISER - FULL PIPELINE TEST"
    )

    print(
        "#" * 70
    )

    # ========================================================
    # VERIFY POSTER
    # ========================================================

    if not os.path.exists(
        POSTER_PATH
    ):

        print()
        print(
            "ERROR: Test poster not found."
        )

        print()
        print(
            "Expected path:"
        )

        print(
            POSTER_PATH
        )

        print()
        print(
            "Please update POSTER_PATH "
            "at the top of this file."
        )

        return

    # ========================================================
    # TEST 1
    # ========================================================

    metadata = test_metadata()

    # ========================================================
    # TEST 2
    # ========================================================

    rag_context = test_rag()

    # ========================================================
    # TEST 3
    # ========================================================

    summary = test_summary(
        metadata
    )

    # ========================================================
    # TEST 4
    # ========================================================

    filtered_students = (
        test_student_filter(
            metadata
        )
    )

    # ========================================================
    # TEST 5
    # ========================================================

    generated_emails = (
        test_email_generation(

            metadata=metadata,

            summary=summary,

            rag_context=rag_context,

            filtered_students=filtered_students
        )
    )

    # ========================================================
    # TEST 6
    # ========================================================

    test_email_attachment(

        generated_emails=
            generated_emails,

        filtered_students=
            filtered_students
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print(
        "#" * 70
    )

    print(
        "FULL PIPELINE TEST COMPLETE"
    )

    print(
        "#" * 70
    )

    print()

    print(
        "Metadata validation       : PASSED"
    )

    print(
        "RAG retrieval             : PASSED"
    )

    print(
        "Summary generation        : PASSED"
    )

    print(
        "Student filtering         : PASSED"
    )

    print(
        "Email generation          : PASSED"
    )

    print(
        "Poster attachment         : PASSED"
    )

    print()

    print(
        "NO REAL EMAILS WERE SENT."
    )

    print()

    print(
        "#" * 70
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_pipeline_test()