from src.extraction.schemas import PosterMetadata


def test_valid_poster_metadata():
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

    assert metadata.title == "FUTURE-ED GLOBAL CONFERENCE"
    assert metadata.university == "CHRIST (Deemed to be University)"
    assert metadata.target_audience == "Researchers and academicians"
    assert metadata.target_audience_type == "Inferred"

    print("\nPosterMetadata validation successful!")
    print(metadata.model_dump_json(indent=4))


if __name__ == "__main__":
    test_valid_poster_metadata()