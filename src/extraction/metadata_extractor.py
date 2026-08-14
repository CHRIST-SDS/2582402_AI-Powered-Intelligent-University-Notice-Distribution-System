import json
import re
from typing import Any, Dict

import ollama


class MetadataExtractor:
    """
    Extracts structured metadata from a university poster by
    combining OCR text and Vision Language Model (VLM) output.

    OCR is treated as the primary source for exact textual
    information, while the VLM is used as a secondary source
    for visual and contextual understanding.

    The extraction is performed using a local Ollama model.
    No external API is used.
    """

    def __init__(
        self,
        model_name: str = "gemma3:latest"
    ):
        self.model_name = model_name

    # ---------------------------------------------------------
    # PROMPT CREATION
    # ---------------------------------------------------------

    def build_prompt(
        self,
        ocr_text: str,
        vlm_output: str
    ) -> str:

        prompt = f"""
You are a reliable university poster metadata extraction
system.

You are given information from TWO sources:

SOURCE 1:
OCR TEXT extracted directly from the poster.

SOURCE 2:
Vision Language Model (VLM) analysis of the poster.

Your task is to extract structured metadata using ONLY
information supported by these two sources.

============================================================
CRITICAL SOURCE-GROUNDING RULE
============================================================

This is a SOURCE-GROUNDED extraction task.

Every factual metadata value must be supported by SOURCE 1
or SOURCE 2.

Do NOT invent, fabricate, or hallucinate:

- names
- guest speakers
- contact persons
- phone numbers
- email addresses
- URLs
- dates
- deadlines
- eligibility requirements
- specific target audiences
- organizations
- event details

Do NOT use general world knowledge to create factual
information.

However, certain fields may use controlled reasoning when
explicitly permitted by their instructions.

These fields are:

- description
- short_summary
- target_audience

Controlled inference MUST be clearly distinguished from
explicitly extracted information.

============================================================
SOURCE PRIORITY
============================================================

OCR is the PRIMARY SOURCE for exact textual information.

The VLM is a SECONDARY SOURCE and should mainly be used for:

- visual context
- poster layout
- section identification
- relationships between text elements
- visual hierarchy
- information OCR may have missed

When OCR and VLM disagree:

1. Prefer clearly readable OCR.

2. Use VLM information only when it is consistent with
   the poster.

3. Do NOT blindly copy VLM information if it conflicts
   with OCR.

4. Do NOT invent information to resolve ambiguity.

============================================================
GENERAL MISSING INFORMATION RULE
============================================================

For normal factual fields:

If information cannot be reliably determined:

"Not Available"

For list fields:

[]

Do NOT use:

- null
- Unknown
- N/A
- Not Mentioned
- Not Clearly Visible
- Unreadable

Use exactly:

"Not Available"

============================================================
UNIVERSITY
============================================================

Prefer the FULL institution name explicitly written on
the poster.

If both an abbreviation and full name exist, use the
full institution name.

Example:

"ACU"
"CHRIST (DEEMED TO BE UNIVERSITY)"

Return:

"CHRIST (DEEMED TO BE UNIVERSITY)"

Do NOT:

- infer a university from a logo alone
- infer a university from an email address
- infer a university from a website domain
- use an abbreviation when the full name is available

If the university cannot be reliably identified:

"Not Available"

============================================================
DEPARTMENT
============================================================

Extract an explicitly named:

- department
- school
- faculty
- centre
- academic unit

Example:

"School of Education"

Do NOT infer a department merely from the event topic.

If none is explicitly supported:

"Not Available"

============================================================
VENUE
============================================================

Extract the physical event location as supported by the
poster.

The venue may contain:

- university name
- campus name
- building
- auditorium
- hall
- conference centre
- room
- other physical location information

Do NOT unnecessarily remove information from the venue.

If the poster explicitly presents:

"CHRIST (Deemed to be University)
Bangalore Central Campus"

as the event location, the venue may be:

"CHRIST (Deemed to be University) Bangalore Central Campus"

If the poster explicitly gives only:

"Bangalore Central Campus"

then use:

"Bangalore Central Campus"

Do not invent or expand the venue.

============================================================
POSTER TYPE
============================================================

Identify the TYPE OF EVENT.

Examples:

- Conference
- Workshop
- Seminar
- Webinar
- Competition
- Hackathon
- Symposium
- Training
- Lecture
- Exhibition
- Recruitment
- Academic Event

Use:

"Conference"

rather than:

"Conference Poster"

when the event is a conference.

If the event type cannot be reliably determined:

"Not Available"

============================================================
TITLE
============================================================

Identify the main official event title using:

- OCR text
- visual hierarchy
- VLM interpretation

Do not invent or embellish the title.

Do not unnecessarily include taglines.

If a generic phrase such as:

"International Conference on"

appears before a distinctive event name, determine whether
it is a descriptor or part of the official title using the
poster structure and VLM context.

Do not make the decision based only on capitalization.

============================================================
DESCRIPTION
============================================================

Create a meaningful description using information contained
in the sources.

The description MAY paraphrase information already present
in the poster.

It MUST NOT introduce unsupported facts.

The description may summarize:

- event purpose
- theme
- focus
- subject
- academic area

Do NOT add unsupported:

- speakers
- audiences
- eligibility
- technologies
- organizations
- registration details
- benefits

If no meaningful description can be created:

"Not Available"

============================================================
SHORT SUMMARY
============================================================

Create a concise one-sentence summary of the poster.

If enough reliable information exists, DO NOT return
"Not Available".

The short summary MAY paraphrase:

- title
- description
- event theme
- event type
- explicitly stated purpose

It MUST NOT introduce new factual information.

Example:

Poster information:

"INTERNATIONAL CONFERENCE ON"
"FUTURE-ED GLOBAL CONFERENCE"

and:

"INTERNATIONAL PERSPECTIVES IN SHAPING LEARNING FOR FUTURE
GENERATIONS"

A valid short summary is:

"An international conference focused on shaping learning
for future generations."

This is allowed because it is a grounded paraphrase.

============================================================
DATE HANDLING
============================================================

Prefer clearly readable OCR dates.

Do not replace OCR dates with guessed VLM dates.

Distinguish:

- event date
- registration deadline
- paper submission deadline
- abstract deadline
- notification date
- publication date
- other important dates

"event_date" contains the main event date.

"registration_deadline" contains a date ONLY when the
poster explicitly identifies it as a registration deadline.

"important_dates" contains other actual dates.

============================================================
IMPORTANT DATES STRICT RULE
============================================================

"important_dates" MUST contain actual date expressions only.

Valid:

- "15 September 2026"
- "15/09/2026"
- "28 & 29 August 2026"
- "Submission deadline: 10 August 2026"

Invalid:

- "Publication Opportunities"
- "Scopus indexed Emerald Journals"
- "Registration Open"
- "Call for Papers"

Never place general information in "important_dates".

If there are no additional dates:

[]

============================================================
REGISTRATION
============================================================

Only extract a registration URL when a URL is explicitly
present and clearly associated with registration.

Do NOT invent URLs.

A QR code does NOT automatically mean that a registration
URL is known.

If the QR destination cannot be determined:

"Not Available"

============================================================
GUEST SPEAKERS
============================================================

A person may ONLY be included in "guest_speakers" when:

1. Their name appears in SOURCE 1 or SOURCE 2.

AND

2. The source explicitly identifies them as a:

- Guest Speaker
- Keynote Speaker
- Invited Speaker
- Chief Guest
- Speaker
- Resource Person
- Featured Speaker
- similar speaking role

NEVER create fictional speaker names.

NEVER create example names.

NEVER infer speakers from the event type.

If no explicitly identified speaker exists:

[]

============================================================
CONTACT PERSON
============================================================

Extract a contact person ONLY when the source explicitly
identifies the person as:

- Contact
- Contact Person
- Coordinator
- Organizer
- Organiser
- Convenor
- Faculty Contact
- Event Contact
- For Queries
- For More Information

A person's name alone does NOT make them a contact person.

A guest speaker is NOT automatically a contact person.

If no explicitly identified contact exists:

"Not Available"

============================================================
CONTACT PHONE
============================================================

Extract phone numbers ONLY when the actual number appears
in SOURCE 1 or SOURCE 2.

NEVER generate phone numbers.

NEVER create example phone numbers.

If no phone number exists:

[]

============================================================
EMAIL
============================================================

Extract email addresses ONLY when present in SOURCE 1
or SOURCE 2.

Do NOT infer email addresses.

Do NOT infer university identity from an email domain.

If no email exists:

"Not Available"

============================================================
ELIGIBILITY
============================================================

Extract eligibility ONLY when explicitly stated or directly
supported by the sources.

Do NOT infer eligibility merely from the event type.

If not explicitly supported:

"Not Available"

============================================================
TARGET AUDIENCE
============================================================

============================================================
TARGET AUDIENCE RULES
============================================================

The target_audience field is a critical field because it will
be used later for filtering posters and sending email
notifications.

Therefore, target audience classification must be conservative
and evidence-based.

1. Set "target_audience_type" to "Explicit" ONLY when the poster
   directly states the intended audience.

   Examples of explicit evidence:

   - "Open to undergraduate students"
   - "For postgraduate students"
   - "Target audience: Teachers and educators"
   - "Eligibility: Faculty members and researchers"
   - "Students, researchers and academicians are invited"

2. Set "target_audience_type" to "Inferred" when the audience
   can reasonably be inferred from the content of the poster,
   but is NOT directly stated.

   Example:

   A research conference discusses academic research but does
   not explicitly state who can attend.

   In this case:

   "target_audience": "Researchers and academicians",
   "target_audience_type": "Inferred"

3. Set "target_audience_type" to "Not Available" when there is
   insufficient evidence to identify the target audience.

4. NEVER infer "Undergraduate and postgraduate students" merely
   because the event is organized by a university.

5. NEVER assume that university students are the target audience
   unless the poster provides evidence.

6. Do NOT use event type alone to determine the target audience.

   For example:

   "Conference" does NOT automatically mean:
   "Undergraduate and postgraduate students".

7. If the poster explicitly lists eligibility criteria, use that
   information to determine the target audience.

8. If multiple audience groups are explicitly mentioned, include
   all of them.

9. The target audience must always be identified.

If the target audience is explicitly stated on the poster,
extract it and set:

"target_audience_type": "Explicit"

If the target audience is not explicitly stated, infer the
most likely target audience using the available evidence from
the OCR text, VLM analysis, event type, department, eligibility,
event description, and subject matter.

In this case, set:

"target_audience_type": "Inferred"

Do NOT return "Not Available" for target_audience.

The target audience should represent the most likely group
that the event is intended for.

10. The target_audience_type field MUST contain exactly one of:

1. Explicit
2. Inferred
3. Default

------------------------------------------------------------
EXPLICIT
------------------------------------------------------------

If the poster explicitly states the intended audience,
extract it.

Examples:

"Open to undergraduate and postgraduate students"

or:

"For teachers, researchers and education professionals"

Then:

"target_audience":
"Undergraduate and postgraduate students"

or:

"target_audience":
"Teachers, researchers and education professionals"

and:

"target_audience_type":
"Explicit"

------------------------------------------------------------
INFERRED
------------------------------------------------------------

If the poster does NOT explicitly state an audience, but
the context provides strong evidence about the likely
audience, a controlled inference MAY be made.

Examples:

A university research conference clearly focused on
academic research may reasonably be targeted toward:

"Students, researchers and faculty"

However, do NOT invent highly specific groups.

Do not infer a specific audience merely because it is
plausible.

If making an inference, set:

"target_audience_type":
"Inferred"

------------------------------------------------------------
DEFAULT
------------------------------------------------------------

If there is no explicit audience and no sufficiently strong
context to identify a narrower audience, use:

"target_audience":
"Open to all"

and:

"target_audience_type":
"Default"

This is a controlled fallback for the dashboard.

"Open to all" means that the system found no reliable
restriction or narrower audience in the available poster
information.

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

Do NOT silently present inferred audience information as
explicitly extracted information.

Always indicate the source type using:

- "Explicit"
- "Inferred"
- "Default"

============================================================
KEYWORDS
============================================================

Keywords must be supported by words or concepts actually
present in the poster.

Do NOT add related concepts that are not supported.

For example, if the poster contains:

Education
Future-Ed
Learning
Conference

these may be keywords.

Do NOT add:

Artificial Intelligence
Machine Learning

unless those concepts are actually supported by SOURCE 1
or SOURCE 2.

Prefer approximately 3 to 8 meaningful keywords.

============================================================
OTHER INFORMATION
============================================================

Use "other_information" for relevant information that does
not naturally belong in another field.

Examples:

- publication opportunities
- journal indexing information
- awards
- announcements
- benefits
- collaboration opportunities
- sponsorship information

Example:

"Publication Opportunities"
"Scopus indexed Emerald Journals"

may be stored under:

"other_information"

Do not place dates here when they can be classified elsewhere.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "poster_type": "Not Available",
    "title": "Not Available",
    "department": "Not Available",
    "university": "Not Available",
    "description": "Not Available",
    "short_summary": "Not Available",
    "event_date": "Not Available",
    "important_dates": [],
    "venue": "Not Available",
    "registration_deadline": "Not Available",
    "registration_link": "Not Available",
    "guest_speakers": [],
    "contact_person": "Not Available",
    "contact_phone": [],
    "email": "Not Available",
    "eligibility": "Not Available",
    "target_audience": "Open to all",
    "target_audience_type": "Default",
    "keywords": [],
    "other_information": "Not Available"
}}

List fields:

- important_dates
- guest_speakers
- contact_phone
- keywords

must always be lists.

============================================================
SOURCE 1 - OCR TEXT
============================================================

{ocr_text}

============================================================
SOURCE 2 - VISION LANGUAGE MODEL OUTPUT
============================================================

{vlm_output}

============================================================
FINAL INSTRUCTION
============================================================

Extract the most reliable metadata from the two sources.

Remember:

OCR = PRIMARY textual evidence
VLM = SECONDARY visual/contextual evidence

Every factual field must be source-grounded.

Descriptions and short summaries may paraphrase existing
information.

Target audience has controlled inference:

- Explicit audience -> "Explicit"
- Strong contextual inference -> "Inferred"
- No reliable audience information -> "Open to all" with
  "Default"

Do NOT fabricate specific target audiences.

Do NOT fabricate names, speakers, phone numbers, contacts,
emails, URLs, dates, eligibility, or other facts.

Return ONLY the JSON object.
"""

        return prompt

    # ---------------------------------------------------------
    # METADATA EXTRACTION
    # ---------------------------------------------------------

    def extract_metadata(
        self,
        ocr_text: str,
        vlm_output: str
    ) -> Dict[str, Any]:

        if not ocr_text.strip() and not vlm_output.strip():
            raise ValueError(
                "Both OCR text and VLM output are empty."
            )

        prompt = self.build_prompt(
            ocr_text=ocr_text,
            vlm_output=vlm_output
        )

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        raw_response = response["message"]["content"]

        return self._parse_json_response(
            raw_response
        )

    # ---------------------------------------------------------
    # JSON PARSING
    # ---------------------------------------------------------

    def _parse_json_response(
        self,
        response_text: str
    ) -> Dict[str, Any]:

        cleaned_response = response_text.strip()

        cleaned_response = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned_response,
            flags=re.IGNORECASE
        )

        cleaned_response = re.sub(
            r"\s*```$",
            "",
            cleaned_response
        )

        cleaned_response = cleaned_response.strip()

        try:

            metadata = json.loads(
                cleaned_response
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "The local LLM did not return valid JSON.\n\n"
                f"Raw response:\n{response_text}"
            ) from error

        if not isinstance(metadata, dict):

            raise ValueError(
                "Expected the LLM response to be a JSON object."
            )

        return self._normalize_metadata(
            metadata
        )

    # ---------------------------------------------------------
    # DATE VALIDATION
    # ---------------------------------------------------------

    def _is_date_like(
        self,
        value: str
    ) -> bool:

        if not isinstance(value, str):
            return False

        text = value.strip()

        if not text:
            return False

        numeric_date = re.search(
            r"\b\d{1,2}\s*[\/\-.]\s*\d{1,2}"
            r"\s*[\/\-.]\s*\d{2,4}\b",
            text
        )

        if numeric_date:
            return True

        month_names = (
            r"January|February|March|April|May|June|July|"
            r"August|September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
        )

        word_date = re.search(
            rf"\b\d{{1,2}}"
            rf"(?:\s*&\s*\d{{1,2}})?"
            rf"\s+(?:{month_names})"
            rf"\s+\d{{4}}\b",
            text,
            flags=re.IGNORECASE
        )

        if word_date:
            return True

        reverse_word_date = re.search(
            rf"\b(?:{month_names})"
            rf"\s+\d{{1,2}}"
            rf"(?:,\s*|\s+)\d{{4}}\b",
            text,
            flags=re.IGNORECASE
        )

        if reverse_word_date:
            return True

        month_year = re.search(
            rf"\b(?:{month_names})\s+\d{{4}}\b",
            text,
            flags=re.IGNORECASE
        )

        if month_year:
            return True

        return False

    # ---------------------------------------------------------
    # METADATA NORMALIZATION
    # ---------------------------------------------------------

    def _normalize_metadata(
        self,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:

        expected_fields = {
            "poster_type": "Not Available",
            "title": "Not Available",
            "department": "Not Available",
            "university": "Not Available",
            "description": "Not Available",
            "short_summary": "Not Available",
            "event_date": "Not Available",
            "important_dates": [],
            "venue": "Not Available",
            "registration_deadline": "Not Available",
            "registration_link": "Not Available",
            "guest_speakers": [],
            "contact_person": "Not Available",
            "contact_phone": [],
            "email": "Not Available",
            "eligibility": "Not Available",
            "target_audience": "Open to all",
            "target_audience_type": "Default",
            "keywords": [],
            "other_information": "Not Available"
        }

        normalized_metadata = {}

        invalid_values = {
            "",
            "null",
            "none",
            "unknown",
            "n/a",
            "not available",
            "not mentioned",
            "not clearly visible",
            "unreadable"
        }

        list_fields = {
            "important_dates",
            "guest_speakers",
            "contact_phone",
            "keywords"
        }

        for field, default_value in expected_fields.items():

            value = metadata.get(
                field,
                default_value
            )

            if value is None:
                value = default_value

            # -------------------------------------------------
            # STRING FIELDS
            # -------------------------------------------------

            if field not in list_fields:

                if not isinstance(value, str):
                    value = str(value)

                value = value.strip()

                if value.lower() in invalid_values:
                    value = default_value

                normalized_metadata[field] = value

                continue

            # -------------------------------------------------
            # LIST FIELDS
            # -------------------------------------------------

            if not isinstance(value, list):
                value = [value]

            cleaned_items = []

            for item in value:

                if item is None:
                    continue

                item = str(item).strip()

                if not item:
                    continue

                if item.lower() in invalid_values:
                    continue

                cleaned_items.append(item)

            normalized_metadata[field] = list(
                dict.fromkeys(cleaned_items)
            )

        # -----------------------------------------------------
        # TARGET AUDIENCE VALIDATION
        # -----------------------------------------------------

        allowed_audience_types = {
            "Explicit",
            "Inferred",
            "Default"
        }

        if (
            normalized_metadata["target_audience_type"]
            not in allowed_audience_types
        ):
            normalized_metadata["target_audience_type"] = "Default"

        if (
            normalized_metadata["target_audience"] == "Not Available"
            or not normalized_metadata["target_audience"]
        ):
            normalized_metadata["target_audience"] = "Open to all"
            normalized_metadata["target_audience_type"] = "Default"

        # -----------------------------------------------------
        # IMPORTANT DATE VALIDATION
        # -----------------------------------------------------

        normalized_metadata["important_dates"] = [
            date
            for date in normalized_metadata["important_dates"]
            if self._is_date_like(date)
        ]

        # -----------------------------------------------------
        # PHONE NUMBER CLEANING
        # -----------------------------------------------------

        valid_phone_numbers = []

        for phone in normalized_metadata["contact_phone"]:

            digits = re.sub(
                r"\D",
                "",
                phone
            )

            if len(digits) >= 7:
                valid_phone_numbers.append(phone)

        normalized_metadata["contact_phone"] = list(
            dict.fromkeys(valid_phone_numbers)
        )

        # -----------------------------------------------------
        # KEYWORD CLEANING
        # -----------------------------------------------------

        cleaned_keywords = []

        for keyword in normalized_metadata["keywords"]:

            keyword = keyword.strip()

            if "@" in keyword:
                continue

            if re.search(
                r"https?://|www\.",
                keyword,
                flags=re.IGNORECASE
            ):
                continue

            if len(keyword.split()) > 6:
                continue

            cleaned_keywords.append(keyword)

        normalized_metadata["keywords"] = list(
            dict.fromkeys(cleaned_keywords)
        )

        # -----------------------------------------------------
        # GUEST SPEAKER CLEANING
        # -----------------------------------------------------

        normalized_metadata["guest_speakers"] = list(
            dict.fromkeys(
                speaker.strip()
                for speaker
                in normalized_metadata["guest_speakers"]
                if speaker.strip()
            )
        )

        return normalized_metadata