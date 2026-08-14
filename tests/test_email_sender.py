import os
import sys
import tempfile

from email import policy
from email.parser import BytesParser


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORT EMAIL SENDER
# ============================================================

from src.email.email_sender import EmailSender


# ============================================================
# CREATE TEMPORARY POSTER
# ============================================================

def create_test_poster():

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    temp_file.write(
        b"%PDF-1.4\n"
        b"% Test poster for email attachment\n"
    )

    temp_file.close()

    return temp_file.name


# ============================================================
# TEST 1: EMAIL MESSAGE CREATION
# ============================================================

def test_message_creation():

    print("=" * 70)
    print("TEST 1: EMAIL MESSAGE CREATION")
    print("=" * 70)

    poster_path = create_test_poster()

    try:

        sender = EmailSender(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="sender@test.com",
            sender_password="test_password"
        )

        message = sender._create_message(

            recipient_email="rahul@email.com",

            recipient_name="Rahul Sharma",

            subject=(
                "Academic Opportunity: "
                "FUTURE-ED GLOBAL CONFERENCE"
            ),

            body="""Dear Rahul Sharma,

Greetings from the School of Education at CHRIST (Deemed to be University).

We are pleased to share information about the FUTURE-ED Global Conference.

Event Details:

Event: FUTURE-ED Global Conference
Date: 28 & 29 August 2026
Venue: CHRIST (Deemed to be University) Bangalore Central Campus
Target Audience: Researchers and academicians

The FUTURE-ED Global Conference is an international conference focused on shaping learning for future generations and exploring international perspectives in education.

Given your academic background in postgraduate Education and your interests in Education, Learning, Research, and Pedagogy, this conference may be relevant to your academic interests.

The original conference poster is attached for your reference.

Warm Regards,
School of Education
CHRIST (Deemed to be University)
""",

            poster_path=poster_path
        )

        print()
        print("From:")
        print(message["From"])

        print()
        print("To:")
        print(message["To"])

        print()
        print("Subject:")
        print(message["Subject"])

        print()
        print("Message created successfully.")

        assert (
            message["To"]
            == "rahul@email.com"
        )

        assert (
            "FUTURE-ED GLOBAL CONFERENCE"
            in message["Subject"]
        )

        print()
        print("TEST 1 PASSED")

    finally:

        if os.path.exists(
            poster_path
        ):

            os.remove(
                poster_path
            )


# ============================================================
# TEST 2: POSTER ATTACHMENT
# ============================================================

def test_poster_attachment():

    print()
    print("=" * 70)
    print("TEST 2: POSTER ATTACHMENT")
    print("=" * 70)

    poster_path = create_test_poster()

    try:

        sender = EmailSender(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="sender@test.com",
            sender_password="test_password"
        )

        message = sender._create_message(

            recipient_email="rahul@email.com",

            recipient_name="Rahul Sharma",

            subject="Test Poster",

            body="This is a test email.",

            poster_path=poster_path
        )

        attachments = []

        for part in message.walk():

            if part.get_content_disposition() == "attachment":

                attachments.append(part)

        print()
        print(
            "Number of attachments:",
            len(attachments)
        )

        assert len(attachments) == 1

        attachment = attachments[0]

        print(
            "Attachment filename:",
            attachment.get_filename()
        )

        print(
            "Attachment content type:",
            attachment.get_content_type()
        )

        assert (
            attachment.get_filename()
            == os.path.basename(
                poster_path
            )
        )

        assert (
            attachment.get_content_type()
            == "application/pdf"
        )

        print()
        print("Poster attachment successfully verified.")

        print()
        print("TEST 2 PASSED")

    finally:

        if os.path.exists(
            poster_path
        ):

            os.remove(
                poster_path
            )


# ============================================================
# TEST 3: EMAIL BODY
# ============================================================

def test_email_body():

    print()
    print("=" * 70)
    print("TEST 3: EMAIL BODY")
    print("=" * 70)

    poster_path = create_test_poster()

    try:

        sender = EmailSender(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="sender@test.com",
            sender_password="test_password"
        )

        body = """Dear Rahul Sharma,

Greetings from the School of Education at CHRIST (Deemed to be University).

We are pleased to share information about the FUTURE-ED Global Conference.

The conference focuses on shaping learning for future generations and exploring international perspectives in education.

Given your academic background in postgraduate Education and your interests in Education, Learning, Research, and Pedagogy, this conference may be relevant to your academic interests.

The original conference poster is attached for your reference.

Warm Regards,
School of Education
CHRIST (Deemed to be University)
"""

        message = sender._create_message(

            recipient_email="rahul@email.com",

            recipient_name="Rahul Sharma",

            subject="FUTURE-ED Global Conference",

            body=body,

            poster_path=poster_path
        )

        message_body = message.get_body(
            preferencelist=("plain",)
        ).get_content()

        print()
        print("Email body successfully extracted.")

        assert (
            "Dear Rahul Sharma"
            in message_body
        )

        assert (
            "FUTURE-ED Global Conference"
            in message_body
        )

        assert (
            "postgraduate Education"
            in message_body
        )

        assert (
            "Education, Learning, Research, and Pedagogy"
            in message_body
        )

        assert (
            "original conference poster"
            in message_body
        )

        print()
        print("Personalization verified.")

        print()
        print("TEST 3 PASSED")

    finally:

        if os.path.exists(
            poster_path
        ):

            os.remove(
                poster_path
            )


# ============================================================
# TEST 4: MULTIPLE EMAIL DATA
# ============================================================

def test_multiple_email_data():

    print()
    print("=" * 70)
    print("TEST 4: MULTIPLE EMAIL DATA")
    print("=" * 70)

    poster_path = create_test_poster()

    try:

        sender = EmailSender(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="sender@test.com",
            sender_password="test_password"
        )

        generated_emails = [

            {
                "student_id": "ST001",

                "name": "Rahul Sharma",

                "email": "rahul@email.com",

                "subject":
                    "Academic Opportunity: "
                    "FUTURE-ED GLOBAL CONFERENCE",

                "body":
                    "Dear Rahul Sharma,\n\n"
                    "This is the generated email.",

                "attachment_path":
                    poster_path
            },

            {
                "student_id": "ST002",

                "name": "Ananya Rao",

                "email": "ananya@email.com",

                "subject":
                    "Academic Opportunity: "
                    "FUTURE-ED GLOBAL CONFERENCE",

                "body":
                    "Dear Ananya Rao,\n\n"
                    "This is the generated email.",

                "attachment_path":
                    poster_path
            }
        ]

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        # We DO NOT call sender.send_emails()
        # because that would attempt to connect to SMTP.
        #
        # Instead we verify every generated email object.
        # ----------------------------------------------------

        for email_data in generated_emails:

            message = sender._create_message(

                recipient_email=
                    email_data["email"],

                recipient_name=
                    email_data["name"],

                subject=
                    email_data["subject"],

                body=
                    email_data["body"],

                poster_path=
                    email_data["attachment_path"]
            )

            assert (
                message["To"]
                == email_data["email"]
            )

            assert (
                email_data["name"]
                in email_data["body"]
            )

            attachments = [

                part
                for part in message.walk()

                if (
                    part.get_content_disposition()
                    == "attachment"
                )
            ]

            assert len(attachments) == 1

        print()
        print(
            f"Successfully validated "
            f"{len(generated_emails)} email objects."
        )

        print()
        print("TEST 4 PASSED")

    finally:

        if os.path.exists(
            poster_path
        ):

            os.remove(
                poster_path
            )


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    test_message_creation()

    test_poster_attachment()

    test_email_body()

    test_multiple_email_data()

    print()
    print("=" * 70)
    print("ALL EMAIL SENDER TESTS PASSED")
    print("=" * 70)