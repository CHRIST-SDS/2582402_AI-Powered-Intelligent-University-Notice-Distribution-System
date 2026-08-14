from src.extraction.schemas import PosterMetadata
from src.vector_store.vector_store import PosterVectorStore
from src.rag.rag_pipeline import PosterRAG


def create_test_poster():

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


def main():

    print("\n========== RAG TEST ==========")

    # Create vector store
    vector_store = PosterVectorStore()

    # Create poster metadata
    poster = create_test_poster()

    # Store only this poster
    vector_store.add_poster(poster)

    # Create RAG pipeline
    rag = PosterRAG(
        vector_store=vector_store,
        model_name="gemma3:latest"
    )

    # -------------------------------
    # Question 1
    # -------------------------------

    question = "Who is the conference intended for?"

    print("\nQUESTION:")
    print(question)

    answer = rag.ask(question)

    print("\nANSWER:")
    print(answer)

    # -------------------------------
    # Question 2
    # -------------------------------

    question = "When is the conference?"

    print("\nQUESTION:")
    print(question)

    answer = rag.ask(question)

    print("\nANSWER:")
    print(answer)

    # -------------------------------
    # Question 3
    # -------------------------------

    question = "What is the registration deadline?"

    print("\nQUESTION:")
    print(question)

    answer = rag.ask(question)

    print("\nANSWER:")
    print(answer)

    # -------------------------------
    # Question 4
    # -------------------------------

    question = "What is the venue?"

    print("\nQUESTION:")
    print(question)

    answer = rag.ask(question)

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    main()