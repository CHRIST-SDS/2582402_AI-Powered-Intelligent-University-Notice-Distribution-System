from src.extraction.schemas import PosterMetadata
from src.vector_store.vector_store import PosterVectorStore


def main():

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

    vector_store = PosterVectorStore()

    vector_store.add_poster(metadata)

    print("\n========== STORED POSTER ==========")

    stored = vector_store.get_current_poster()

    print(stored)

    print("\n========== RAG SEARCH TEST ==========")

    results = vector_store.search(
        "Who is the conference intended for?"
    )

    print(results)


if __name__ == "__main__":
    main()