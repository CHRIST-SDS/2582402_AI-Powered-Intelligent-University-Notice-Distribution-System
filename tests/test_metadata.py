import json
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.metadata_extractor import MetadataExtractor


def main():
    """
    Test the MetadataExtractor using sample OCR and VLM output.
    """

    sample_ocr = """
    ACU
    CHRIST
    SCHOOL OF EDUCATION

    INTERNATIONAL CONFERENCE ON
    FUTURE-ED GLOBAL CONFERENCE

    INTERNATIONAL PERSPECTIVES
    IN SHAPING LEARNING FOR FUTURE
    GENERATIONS

    28 & 29 AUGUST 2026

    CHRIST (DEEMED TO BE UNIVERSITY)
    Bangalore Central Campus

    Publication Opportunities
    Scopus indexed Emerald Journals
    """

    sample_vlm = """
    The poster appears to be an academic conference poster.

    It contains a university logo, conference information,
    an event title, an event theme, an event date, a campus
    location, a QR code and publication information.

    The main event appears to be an international conference
    related to education and future learning.

    The poster appears to be associated with the School of
    Education at CHRIST (Deemed to be University).

    The event is associated with the Bangalore Central Campus.

    The poster mentions publication opportunities through
    Scopus indexed Emerald Journals.

    The main theme focuses on international perspectives in
    shaping learning for future generations.

    Some text is difficult to read.
    """

    print("\n==========================================")
    print("       METADATA EXTRACTOR TEST")
    print("==========================================\n")

    print("Loading Gemma 3 metadata extractor...")

    extractor = MetadataExtractor(
        model_name="gemma3:latest"
    )

    print("Extractor initialized successfully.")
    print("Sending OCR text and VLM output to Gemma 3...\n")

    metadata = extractor.extract_metadata(
        ocr_text=sample_ocr,
        vlm_output=sample_vlm
    )

    print("========== EXTRACTED METADATA ==========\n")

    print(
        json.dumps(
            metadata,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\n==========================================")
    print("             TEST COMPLETED")
    print("==========================================\n")


if __name__ == "__main__":
    main()