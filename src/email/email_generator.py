import ollama

from src.extraction.schemas import PosterMetadata


class EmailGenerator:
    """
    Generates factual, personalized academic emails.

    Sources:
        - Validated PosterMetadata
        - Generated poster summary
        - Retrieved RAG context
        - Selected student profile

    Gemma 3 is used for natural-language generation only.

    Student filtering/eligibility is handled separately.
    """

    def __init__(self, model_name="gemma3:latest"):
        self.model_name = model_name

    # ==========================================================
    # CHECK WHETHER VALUE IS AVAILABLE
    # ==========================================================

    @staticmethod
    def _is_available(value):

        if value is None:
            return False

        if isinstance(value, str):

            value = value.strip().lower()

            unavailable_values = {
                "",
                "not available",
                "n/a",
                "na",
                "unknown",
                "none",
                "null"
            }

            return value not in unavailable_values

        if isinstance(value, list):
            return len(value) > 0

        return True

    # ==========================================================
    # BUILD POSTER CONTEXT
    # ==========================================================

    def _build_poster_context(
        self,
        metadata: PosterMetadata,
        summary: str,
        rag_context: str = ""
    ):

        important_dates = (
            "\n".join(metadata.important_dates)
            if metadata.important_dates
            else "Not Available"
        )

        keywords = (
            ", ".join(metadata.keywords)
            if metadata.keywords
            else "Not Available"
        )

        guest_speakers = (
            ", ".join(metadata.guest_speakers)
            if metadata.guest_speakers
            else "Not Available"
        )

        contact_phone = (
            ", ".join(metadata.contact_phone)
            if metadata.contact_phone
            else "Not Available"
        )

        context = f"""
VALIDATED POSTER METADATA
=========================

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
{important_dates}

Venue:
{metadata.venue}

Registration Deadline:
{metadata.registration_deadline}

Registration Link:
{metadata.registration_link}

Guest Speakers:
{guest_speakers}

Contact Person:
{metadata.contact_person}

Contact Phone:
{contact_phone}

Email:
{metadata.email}

Eligibility:
{metadata.eligibility}

Target Audience:
{metadata.target_audience}

Target Audience Type:
{metadata.target_audience_type}

Keywords:
{keywords}

Other Information:
{metadata.other_information}


VALIDATED POSTER SUMMARY
========================

{summary}


RETRIEVED RAG CONTEXT
=====================

{rag_context if rag_context else "No additional RAG context available."}
"""

        return context.strip()

    # ==========================================================
    # BUILD STUDENT CONTEXT
    # ==========================================================

    def _build_student_context(self, student):

        return f"""
SELECTED STUDENT INFORMATION
============================

Name:
{student.get("name", "Student")}

Programme:
{student.get("programme", "Not Available")}

Department:
{student.get("department", "Not Available")}

Level:
{student.get("level", "Not Available")}

Interests:
{student.get("interests", "Not Available")}
""".strip()

    # ==========================================================
    # GENERATE EMAIL
    # ==========================================================

    def generate_email(
        self,
        metadata: PosterMetadata,
        summary: str,
        student,
        rag_context: str = ""
    ):

        poster_context = self._build_poster_context(
            metadata=metadata,
            summary=summary,
            rag_context=rag_context
        )

        student_context = self._build_student_context(
            student=student
        )

        prompt = f"""
You are an academic communication assistant for
CHRIST (Deemed to be University).

Write a professional, factual and personalized academic
email to the selected student about the event represented
by the uploaded poster.

The student has ALREADY been selected by a separate
student-filtering system.

Do not make eligibility decisions yourself.

Your responsibility is to communicate the verified
event information clearly and explain why the event
may be relevant to this particular student's academic
background and interests.

==========================================================
STRICT FACTUALITY
==========================================================

Use ONLY:

1. Validated PosterMetadata
2. Validated Poster Summary
3. Retrieved RAG Context
4. Student Information

Never invent information.

Never guess missing information.

Never introduce information from general knowledge.

Never modify dates, deadlines, links, venues, eligibility,
publication information or other factual details.

==========================================================
MISSING INFORMATION
==========================================================

This is a student-facing email.

If information is:

- Not Available
- N/A
- NA
- Unknown
- None
- Null
- Empty
- Empty list

DO NOT mention that it is missing.

DO NOT write:

"The registration link is not available."

DO NOT write:

"The poster does not mention a deadline."

DO NOT write:

"No contact information was provided."

Simply omit the unavailable information.

==========================================================
SUMMARY LENGTH
==========================================================

The event summary in the email must NOT be overly short.

Write approximately 2-3 paragraphs containing useful
factual information from the poster.

The summary should explain:

1. What the event is.
2. What the event focuses on.
3. Who the event is intended for.
4. Important academic opportunities explicitly mentioned.
5. Other relevant information available in the
   validated metadata or RAG context.

Do NOT simply repeat the Event Details section.

The summary should add useful context.

Do not make the summary excessively long.

Aim for approximately 100-160 words when enough
verified information is available.

If the poster contains limited information, use
the available information without inventing details.

==========================================================
ACADEMIC PERSONALIZATION
==========================================================

The email MUST include a factual personalization
section whenever sufficient student information exists.

Use the student's:

- Programme
- Department
- Level
- Interests

to explain the relevance of the event.

The personalization must be based on actual information
from the student's profile.

For example:

"Given your academic background in postgraduate Education
and your interests in Education, Learning, Research, and
Pedagogy, this conference may be relevant to your academic
interests."

Another acceptable example:

"Your postgraduate studies in Education and interests in
Learning and Research align with the educational focus
of this conference."

Do NOT invent academic achievements.

Do NOT claim the student has attended previous events.

Do NOT claim the student is an expert.

Do NOT claim guaranteed benefits.

Do NOT say:

"This conference is perfect for you."

Do NOT say:

"This conference will definitely benefit your career."

Use conservative wording such as:

"may be relevant to your academic interests."

==========================================================
PROMOTIONAL LANGUAGE
==========================================================

Avoid unsupported promotional claims such as:

- prestigious
- renowned
- leading
- exclusive
- highly valuable
- career-changing
- once-in-a-lifetime
- leading thinkers
- top researchers

unless explicitly supported by the source.

==========================================================
INTERNAL INFORMATION
==========================================================

NEVER mention:

- Relevance score
- Selection score
- Selection reason
- Filtering algorithm
- Student ranking
- AI
- Gemma
- RAG
- Metadata extraction
- Internal database
- Internal system decisions

==========================================================
EVENT DETAILS
==========================================================

Include the following when available:

Event
Date
Venue
Target Audience

If unavailable, omit the field.

==========================================================
IMPORTANT DATES
==========================================================

Include important dates only when they are available.

Do not invent deadlines.

Do not reinterpret dates.

==========================================================
REGISTRATION
==========================================================

If registration deadline exists:

Include it.

If registration link exists:

Include the exact link.

If either is unavailable:

Omit it completely.

Never tell the student that it is unavailable.

==========================================================
PUBLICATION INFORMATION
==========================================================

Include publication opportunities ONLY if explicitly
supported by the poster metadata or RAG context.

Do not exaggerate publication opportunities.

For example:

"Publication opportunities, including access to
Scopus indexed Emerald Journals, are mentioned
on the poster."

Do not change this into:

"Your paper will be published in a Scopus journal."

==========================================================
ATTACHMENT
==========================================================

The original uploaded poster will be attached separately
by the email-sending module.

Include:

"The original conference poster is attached for
your reference."

==========================================================
EMAIL STRUCTURE
==========================================================

Use this structure:

Dear <student name>,

Greetings from <department> at <university>.

We are pleased to share information about
<event title>.

Event Details:

Event:
Date:
Venue:
Target Audience:

Then provide a meaningful 2-3 paragraph factual
summary of the event.

Then include Important Dates if available.

Then include Registration information if available.

Then include other verified information if relevant.

Then include a personalized paragraph explaining the
connection between the student's academic background,
programme, department and interests and the event.

Then:

"The original conference poster is attached for
your reference."

Warm Regards,
<department>
<university>

==========================================================
POSTER INFORMATION
==========================================================

{poster_context}

==========================================================
STUDENT INFORMATION
==========================================================

{student_context}

==========================================================
FINAL REQUIREMENTS
==========================================================

Return ONLY the final email body.

Do not return analysis.

Do not return JSON.

Do not return a subject line.

Do not mention missing information.

Do not invent information.

Do not make unsupported claims.

Make the summary informative but concise.

Include academic personalization.

Use factual and professional university communication.
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.1
            }
        )

        return response["message"]["content"].strip()

    # ==========================================================
    # GENERATE EMAILS FOR MULTIPLE STUDENTS
    # ==========================================================

    def generate_emails(
        self,
        metadata: PosterMetadata,
        summary: str,
        students,
        rag_context: str = "",
        poster_path: str = None
    ):

        generated_emails = []

        for _, student in students.iterrows():

            email_content = self.generate_email(
                metadata=metadata,
                summary=summary,
                student=student.to_dict(),
                rag_context=rag_context
            )

            generated_emails.append(
                {
                    "student_id": student["student_id"],
                    "name": student["name"],
                    "email": student["email"],
                    "subject": (
                        f"Academic Opportunity: "
                        f"{metadata.title}"
                    ),
                    "body": email_content,
                    "attachment_path": poster_path
                }
            )

        return generated_emails