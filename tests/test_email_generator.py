import sys
import os
import pandas as pd

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.email.email_generator import EmailGenerator
from src.extraction.schemas import PosterMetadata


# ==========================================================
# POSTER METADATA
# ==========================================================

metadata = PosterMetadata(

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
        "An international conference focused on shaping "
        "learning for future generations."
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


# ==========================================================
# GENERATED SUMMARY
# ==========================================================

summary = """
The School of Education at CHRIST (Deemed to be University)
is organizing the FUTURE-ED Global Conference on 28 & 29
August 2026 at the CHRIST (Deemed to be University)
Bangalore Central Campus.

This international conference focuses on shaping learning
for future generations and exploring international
perspectives in education.

The event is intended for researchers and academicians.
Publication opportunities, including access to Scopus
indexed Emerald Journals, are available.

Registration details and a registration link are not
available on the poster.
""".strip()


# ==========================================================
# STUDENT
# ==========================================================

students = pd.DataFrame([

    {
        "student_id": "ST001",
        "name": "Rahul Sharma",
        "email": "rahul@email.com",
        "programme": "MSc Education",
        "department": "School of Education",
        "level": "Postgraduate",
        "interests": (
            "Education, Learning, Research, Pedagogy"
        ),
        "relevance_score": 18
    }

])


# ==========================================================
# RAG CONTEXT
# ==========================================================

rag_context = """
Poster Type: Conference

Title: FUTURE-ED GLOBAL CONFERENCE

Department: School of Education

University: CHRIST (Deemed to be University)

Event Date: 28 & 29 August 2026

Venue: CHRIST (Deemed to be University)
Bangalore Central Campus

Target Audience: Researchers and academicians

Other Information:
Publication Opportunities, Scopus indexed Emerald Journals
""".strip()


# ==========================================================
# CREATE EMAIL GENERATOR
# ==========================================================

generator = EmailGenerator(
    model_name="gemma3:latest"
)


# ==========================================================
# GENERATE EMAIL
# ==========================================================

print("\n")
print("=" * 70)
print("GENERATING EMAIL")
print("=" * 70)


emails = generator.generate_emails(
    metadata=metadata,
    summary=summary,
    students=students,
    rag_context=rag_context
)


# ==========================================================
# DISPLAY EMAIL
# ==========================================================

for email in emails:

    print("\n")
    print("=" * 70)
    print("STUDENT")
    print("=" * 70)

    print(
        f"Name: {email['name']}"
    )

    print(
        f"Email: {email['email']}"
    )

    print(
        f"Subject: {email['subject']}"
    )

    print("\n")
    print("=" * 70)
    print("GENERATED EMAIL")
    print("=" * 70)

    print(
        email["body"]
    )


print("\n")
print("=" * 70)
print("EMAIL GENERATION TEST COMPLETE")
print("=" * 70)