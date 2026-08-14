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

from src.student_filtering.student_filter import StudentFilter
from src.extraction.schemas import PosterMetadata


# ==========================================================
# STUDENT DATA
# ==========================================================

students = pd.DataFrame([

    {
        "student_id": "ST001",
        "name": "Rahul Sharma",
        "email": "rahul@email.com",
        "programme": "MSc Education",
        "department": "School of Education",
        "level": "Postgraduate",
        "interests": "Education, Learning, Research, Pedagogy"
    },

    {
        "student_id": "ST002",
        "name": "Ananya Rao",
        "email": "ananya@email.com",
        "programme": "BSc Psychology",
        "department": "Psychology",
        "level": "Undergraduate",
        "interests": "Education, Learning"
    },

    {
        "student_id": "ST003",
        "name": "Arjun Kumar",
        "email": "arjun@email.com",
        "programme": "MSc Data Science",
        "department": "Computer Science",
        "level": "Postgraduate",
        "interests": "AI, Machine Learning, Data Science"
    },

    {
        "student_id": "ST004",
        "name": "Priya Nair",
        "email": "priya@email.com",
        "programme": "PhD Education",
        "department": "School of Education",
        "level": "Doctoral",
        "interests": "Education, Research, Pedagogy"
    }

])


filter_engine = StudentFilter(students)


# ==========================================================
# TEST 1
# EXPLICIT PHD AUDIENCE
# ==========================================================

print("\n")
print("=" * 60)
print("TEST 1: EXPLICIT PHD AUDIENCE")
print("=" * 60)


phd_metadata = PosterMetadata(

    poster_type="Conference",

    title="ADVANCED RESEARCH CONFERENCE",

    department="School of Education",

    university="CHRIST (Deemed to be University)",

    description=(
        "Conference for doctoral researchers "
        "and PhD scholars."
    ),

    short_summary=(
        "Conference specifically intended "
        "for PhD scholars."
    ),

    event_date="20 October 2026",

    important_dates=[
        "20 October 2026"
    ],

    venue="Bangalore Central Campus",

    registration_deadline="Not Available",

    registration_link="Not Available",

    guest_speakers=[],

    contact_person="Not Available",

    contact_phone=[],

    email="Not Available",

    eligibility="PhD Scholars only",

    target_audience="PhD Scholars",

    target_audience_type="Explicit",

    keywords=[
        "Education",
        "Research",
        "Academia"
    ],

    other_information="Research presentations"
)


phd_results = filter_engine.filter_students(
    phd_metadata,
    minimum_score=5
)


print("\nEligible students:")

if phd_results.empty:

    print("No students selected.")

else:

    print(
        phd_results[
            [
                "student_id",
                "name",
                "level",
                "relevance_score",
                "selection_reason"
            ]
        ].to_string(index=False)
    )


# ==========================================================
# TEST 2
# EXPLICIT UNDERGRADUATE + POSTGRADUATE
# ==========================================================

print("\n")
print("=" * 60)
print("TEST 2: EXPLICIT UNDERGRADUATE + POSTGRADUATE")
print("=" * 60)


student_metadata = PosterMetadata(

    poster_type="Workshop",

    title="FUTURE LEARNING WORKSHOP",

    department="School of Education",

    university="CHRIST (Deemed to be University)",

    description=(
        "Workshop designed for undergraduate "
        "and postgraduate students."
    ),

    short_summary=(
        "Workshop focused on future learning."
    ),

    event_date="15 September 2026",

    important_dates=[
        "15 September 2026"
    ],

    venue="Bangalore Central Campus",

    registration_deadline="Not Available",

    registration_link="Not Available",

    guest_speakers=[],

    contact_person="Not Available",

    contact_phone=[],

    email="Not Available",

    eligibility=(
        "Undergraduate and postgraduate students"
    ),

    target_audience=(
        "Undergraduate and postgraduate students"
    ),

    target_audience_type="Explicit",

    keywords=[
        "Education",
        "Learning",
        "Future"
    ],

    other_information="Student workshop"
)


student_results = filter_engine.filter_students(
    student_metadata,
    minimum_score=5
)


print("\nEligible students:")

if student_results.empty:

    print("No students selected.")

else:

    print(
        student_results[
            [
                "student_id",
                "name",
                "level",
                "relevance_score",
                "selection_reason"
            ]
        ].to_string(index=False)
    )


# ==========================================================
# TEST 3
# INFERRED RESEARCHERS AND ACADEMICIANS
# ==========================================================

print("\n")
print("=" * 60)
print("TEST 3: INFERRED RESEARCHERS AND ACADEMICIANS")
print("=" * 60)


research_metadata = PosterMetadata(

    poster_type="Conference",

    title="FUTURE-ED GLOBAL CONFERENCE",

    department="School of Education",

    university="CHRIST (Deemed to be University)",

    description=(
        "International conference focused on "
        "shaping learning for future generations."
    ),

    short_summary=(
        "An international conference focused "
        "on shaping learning for future generations."
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

    target_audience=(
        "Researchers and academicians"
    ),

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


research_results = filter_engine.filter_students(
    research_metadata,
    minimum_score=5
)


print("\nPotentially relevant students:")

if research_results.empty:

    print("No students selected.")

else:

    print(
        research_results[
            [
                "student_id",
                "name",
                "level",
                "relevance_score",
                "selection_reason"
            ]
        ].to_string(index=False)
    )


# ==========================================================
# TEST COMPLETE
# ==========================================================

print("\n")
print("=" * 60)
print("ALL FILTERING TESTS COMPLETE")
print("=" * 60)