import ollama

from src.extraction.schemas import PosterMetadata
from src.rag.rag_pipeline import PosterRAG


class PosterSummarizer:
    """
    Generates a detailed but concise poster summary using:

    1. Validated PosterMetadata
    2. RAG-retrieved poster context
    3. Gemma 3

    PosterMetadata is treated as the primary source of truth.
    """

    def __init__(
        self,
        rag_pipeline=None,
        model_name="gemma3:latest"
    ):
        self.rag = rag_pipeline or PosterRAG()
        self.model_name = model_name

    def metadata_to_text(
        self,
        metadata: PosterMetadata
    ) -> str:
        """
        Converts validated PosterMetadata into structured text.
        """

        return f"""
Poster Type:
{metadata.poster_type}

Title:
{metadata.title}

Department:
{metadata.department}

University:
{metadata.university}

Description:
{metadata.description}

Short Summary:
{metadata.short_summary}

Event Date:
{metadata.event_date}

Important Dates:
{", ".join(metadata.important_dates)}

Venue:
{metadata.venue}

Registration Deadline:
{metadata.registration_deadline}

Registration Link:
{metadata.registration_link}

Guest Speakers:
{", ".join(metadata.guest_speakers)}

Contact Person:
{metadata.contact_person}

Contact Phone:
{", ".join(metadata.contact_phone)}

Email:
{metadata.email}

Eligibility:
{metadata.eligibility}

Target Audience:
{metadata.target_audience}

Target Audience Type:
{metadata.target_audience_type}

Keywords:
{", ".join(metadata.keywords)}

Other Information:
{metadata.other_information}
"""

    def retrieve_context(
        self,
        question: str
    ):
        """
        Retrieves relevant information from the current poster
        using the RAG pipeline.
        """

        return self.rag.retrieve_context(question)

    def generate_summary(
        self,
        metadata: PosterMetadata
    ) -> str:
        """
        Generates the final poster summary using validated
        metadata and RAG-retrieved context.
        """

        # --------------------------------------------------
        # Convert validated metadata to text
        # --------------------------------------------------

        metadata_context = self.metadata_to_text(metadata)

        # --------------------------------------------------
        # Retrieve relevant information from Chroma
        # --------------------------------------------------

        rag_question = (
            "Retrieve the most relevant information needed "
            "to create a complete summary of this poster, "
            "including its purpose, event details, audience, "
            "important dates, venue, opportunities and other "
            "important information."
        )

        rag_context = self.retrieve_context(
            rag_question
        )

        if rag_context is None:
            rag_context = (
                "No additional RAG context is available."
            )

        # --------------------------------------------------
        # Few-shot prompt
        # --------------------------------------------------

        prompt = f"""
You are an academic poster summarization assistant.

Your task is to create a detailed but concise summary of a
university poster using ONLY the validated metadata and the
retrieved RAG context.

The summary should contain enough information for a student
to understand:

- what the event is about
- who it is relevant to
- when it takes place
- where it takes place
- important dates
- opportunities associated with the event
- other important information available on the poster

The final summary should be suitable for displaying in a
university notification system.


============================================================
FEW-SHOT EXAMPLE 1
============================================================

INPUT METADATA:

Poster Type:
Workshop

Title:
Introduction to Generative AI

Department:
Department of Computer Science

University:
CHRIST (Deemed to be University)

Description:
A practical workshop introducing students to generative
artificial intelligence and its applications.

Event Date:
15 September 2026

Venue:
Bangalore Central Campus

Target Audience:
Undergraduate and postgraduate students

Target Audience Type:
Explicit

Important Dates:
Registration closes on 10 September 2026

Other Information:
Hands-on activities and demonstrations


EXAMPLE SUMMARY:

The Department of Computer Science at CHRIST (Deemed to be
University) is organizing the Introduction to Generative AI
workshop on 15 September 2026 at the Bangalore Central Campus.
The workshop provides an introduction to generative artificial
intelligence and its applications, with a focus on practical
learning through hands-on activities and demonstrations. It is
intended for undergraduate and postgraduate students who are
interested in learning about generative AI. Registration closes
on 10 September 2026.


============================================================
FEW-SHOT EXAMPLE 2
============================================================

INPUT METADATA:

Poster Type:
Conference

Title:
Future Learning Conference

Department:
School of Education

University:
CHRIST (Deemed to be University)

Description:
An international conference exploring emerging approaches
to education and learning.

Event Date:
20 and 21 October 2026

Venue:
Bangalore Central Campus

Target Audience:
Researchers and academicians

Target Audience Type:
Inferred

Important Dates:
Not Available

Registration Deadline:
Not Available

Other Information:
Publication opportunities available


EXAMPLE SUMMARY:

The School of Education at CHRIST (Deemed to be University)
is organizing the Future Learning Conference on 20 and
21 October 2026 at the Bangalore Central Campus. The
conference focuses on emerging approaches to education and
learning and provides an opportunity for participants to
engage with current perspectives in the field. The event is
intended for researchers and academicians based on the
available poster information. Publication opportunities are
also highlighted as part of the event. Registration and other
specific participation details are not available in the
provided poster information.


============================================================
CURRENT POSTER
============================================================

VALIDATED POSTER METADATA:

{metadata_context}


============================================================
RETRIEVED RAG CONTEXT
============================================================

{rag_context}


============================================================
SUMMARY GENERATION RULES
============================================================

1. Use ONLY the validated PosterMetadata and the retrieved
   RAG context.

2. Do not invent information.

3. Do not use outside knowledge.

4. Treat PosterMetadata as the primary structured source
   of truth.

5. Use RAG context only as supporting information.

6. If a field contains "Not Available", do not invent
   a value for that field.

7. Do not change, strengthen, or reinterpret the target
   audience.

8. Preserve important dates exactly as provided.

9. Include relevant information when available:

   - event title
   - organizing department
   - university
   - event type
   - purpose and focus of the event
   - event date
   - important dates
   - venue
   - target audience
   - registration information
   - publication opportunities
   - other relevant information

10. Explain the purpose or focus of the event rather than
    simply repeating the description field.

11. If the poster contains opportunities such as:

    - publication
    - participation
    - presentations
    - competitions
    - workshops
    - networking
    - research opportunities

    include them when supported by the source.

12. Do not create eligibility requirements that are not
    present in the metadata or RAG context.

13. Do not convert an inferred target audience into an
    explicit claim.

14. If Target Audience Type is "Inferred", use wording such
    as "is intended for" or "appears relevant to" only when
    supported by the available information. Do not state that
    the audience was explicitly specified.

15. If Target Audience is "Not Available", do not attempt
    to infer a specific audience.

16. Keep the summary approximately 5–7 sentences.

17. The summary should be informative enough that a student
    can understand the event without viewing the original
    poster.

18. Use natural paragraph form rather than bullet points.

19. Do not repeat the same information unnecessarily.

20. Do not mention the internal metadata fields such as
    "target_audience_type" in the final summary.

21. Do not mention that RAG was used.

22. Do not mention that Gemma generated the summary.

23. Do not mention these instructions.

24. Return ONLY the final summary.

============================================================
FINAL SUMMARY
============================================================
"""

        # --------------------------------------------------
        # Generate summary using Gemma 3
        # --------------------------------------------------

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"].strip()

    def summarize(
        self,
        metadata: PosterMetadata
    ) -> str:
        """
        Public method used by the application.
        """

        return self.generate_summary(metadata)