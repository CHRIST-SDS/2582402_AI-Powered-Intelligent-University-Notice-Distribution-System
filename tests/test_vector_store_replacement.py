from src.extraction.schemas import PosterMetadata
from src.vector_store.vector_store import PosterVectorStore


def create_poster_a():
    return PosterMetadata(
        poster_type="Conference",
        title="FUTURE-ED GLOBAL CONFERENCE",
        department="School of Education",
        university="CHRIST (Deemed to be University)",
        description="Conference focused on shaping learning for future generations.",
        short_summary="International education conference.",
        event_date="28 & 29 August 2026",
        important_dates=["28 & 29 August 2026"],
        venue="CHRIST Bangalore Central Campus",
        registration_deadline="Not Available",
        registration_link="Not Available",
        guest_speakers=[],
        contact_person="Not Available",
        contact_phone=[],
        email="Not Available",
        eligibility="Not Available",
        target_audience="Researchers and academicians",
        target_audience_type="Inferred",
        keywords=["Education", "Future-Ed", "Learning"],
        other_information="Scopus indexed Emerald Journals"
    )


def create_poster_b():
    return PosterMetadata(
        poster_type="Workshop",
        title="ARTIFICIAL INTELLIGENCE WORKSHOP",
        department="Department of Computer Science",
        university="CHRIST (Deemed to be University)",
        description="Workshop focused on artificial intelligence and machine learning.",
        short_summary="AI and machine learning workshop.",
        event_date="15 September 2026",
        important_dates=["10 September 2026"],
        venue="CHRIST Bangalore Central Campus",
        registration_deadline="10 September 2026",
        registration_link="Not Available",
        guest_speakers=[],
        contact_person="Not Available",
        contact_phone=[],
        email="Not Available",
        eligibility="Students interested in artificial intelligence.",
        target_audience="Computer Science students",
        target_audience_type="Explicit",
        keywords=["Artificial Intelligence", "Machine Learning", "Workshop"],
        other_information="Hands-on workshop"
    )


def main():

    vector_store = PosterVectorStore()

    # -------------------------------
    # Store Poster A
    # -------------------------------

    poster_a = create_poster_a()

    vector_store.add_poster(poster_a)

    stored_a = vector_store.get_current_poster()

    print("\n========== AFTER POSTER A ==========")
    print(stored_a["ids"])
    print(stored_a["metadatas"])

    assert stored_a["ids"] == ["current_poster"]

    assert (
        stored_a["metadatas"][0]["title"]
        == "FUTURE-ED GLOBAL CONFERENCE"
    )

    # -------------------------------
    # Store Poster B
    # -------------------------------

    poster_b = create_poster_b()

    vector_store.add_poster(poster_b)

    stored_b = vector_store.get_current_poster()

    print("\n========== AFTER POSTER B ==========")
    print(stored_b["ids"])
    print(stored_b["metadatas"])

    # Only one poster should exist
    assert stored_b["ids"] == ["current_poster"]

    # Poster A must no longer exist
    assert (
        stored_b["metadatas"][0]["title"]
        != "FUTURE-ED GLOBAL CONFERENCE"
    )

    # Poster B must be the current poster
    assert (
        stored_b["metadatas"][0]["title"]
        == "ARTIFICIAL INTELLIGENCE WORKSHOP"
    )

    assert (
        stored_b["metadatas"][0]["target_audience"]
        == "Computer Science students"
    )

    print("\nSingle-poster replacement test PASSED!")


if __name__ == "__main__":
    main()