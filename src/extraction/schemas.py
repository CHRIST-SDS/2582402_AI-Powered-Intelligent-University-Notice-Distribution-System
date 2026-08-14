from typing import List, Literal

from pydantic import BaseModel, Field


class PosterMetadata(BaseModel):
    """
    Pydantic schema for validated poster metadata.

    The target_audience field is mandatory because it is used
    later for Streamlit filtering and email notifications.
    """

    poster_type: str = Field(
        description="Type of poster or event."
    )

    title: str = Field(
        description="Title of the event or poster."
    )

    department: str = Field(
        description="Department organizing the event."
    )

    university: str = Field(
        description="University or organization hosting the event."
    )

    description: str = Field(
        description="Detailed description of the event."
    )

    short_summary: str = Field(
        description="Short summary of the event."
    )

    event_date: str = Field(
        description="Date or date range of the event."
    )

    important_dates: List[str] = Field(
        default_factory=list,
        description="Other important dates mentioned on the poster."
    )

    venue: str = Field(
        description="Location or venue of the event."
    )

    registration_deadline: str = Field(
        description="Registration deadline."
    )

    registration_link: str = Field(
        description="Registration URL or website."
    )

    guest_speakers: List[str] = Field(
        default_factory=list,
        description="Guest speakers mentioned on the poster."
    )

    contact_person: str = Field(
        description="Name of the contact person."
    )

    contact_phone: List[str] = Field(
        default_factory=list,
        description="Contact phone numbers."
    )

    email: str = Field(
        description="Contact email address."
    )

    eligibility: str = Field(
        description="Eligibility requirements for the event."
    )

    target_audience: str = Field(
        min_length=1,
        description=(
            "Target audience of the event. This field must "
            "always contain the most appropriate audience."
        )
    )

    target_audience_type: Literal[
        "Explicit",
        "Inferred"
    ] = Field(
        description=(
            "Whether the target audience was explicitly stated "
            "or inferred from the poster."
        )
    )

    keywords: List[str] = Field(
        default_factory=list,
        description="Important keywords associated with the poster."
    )

    other_information: str = Field(
        description="Other relevant information from the poster."
    )