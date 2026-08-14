import os
import smtplib

from email.message import EmailMessage
from email.utils import formataddr


class EmailSender:
    """
    Sends generated student emails with the original
    uploaded poster attached.

    The email content is generated separately by
    EmailGenerator.

    This class is responsible only for:
        - Creating the email
        - Adding the poster attachment
        - Sending the email
        - Returning the sending status
    """

    def __init__(
        self,
        smtp_server,
        smtp_port,
        sender_email,
        sender_password,
        sender_name="CHRIST (Deemed to be University)"
    ):

        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.sender_name = sender_name

    # ==========================================================
    # ATTACH POSTER
    # ==========================================================

    def _attach_poster(
        self,
        message,
        poster_path
    ):
        """
        Attach the exact poster uploaded by the user.
        """

        if not poster_path:
            return

        if not os.path.exists(poster_path):

            raise FileNotFoundError(
                f"Poster file not found: {poster_path}"
            )

        filename = os.path.basename(
            poster_path
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        # ------------------------------------------------------
        # PDF
        # ------------------------------------------------------

        if extension == ".pdf":

            maintype = "application"
            subtype = "pdf"

        # ------------------------------------------------------
        # PNG
        # ------------------------------------------------------

        elif extension == ".png":

            maintype = "image"
            subtype = "png"

        # ------------------------------------------------------
        # JPG / JPEG
        # ------------------------------------------------------

        elif extension in [".jpg", ".jpeg"]:

            maintype = "image"
            subtype = "jpeg"

        # ------------------------------------------------------
        # OTHER FILE TYPES
        # ------------------------------------------------------

        else:

            maintype = "application"
            subtype = "octet-stream"

        # ------------------------------------------------------
        # READ ORIGINAL FILE
        # ------------------------------------------------------

        with open(
            poster_path,
            "rb"
        ) as file:

            file_data = file.read()

        # ------------------------------------------------------
        # ADD ATTACHMENT
        # ------------------------------------------------------

        message.add_attachment(
            file_data,
            maintype=maintype,
            subtype=subtype,
            filename=filename
        )

    # ==========================================================
    # CREATE EMAIL
    # ==========================================================

    def _create_message(
        self,
        recipient_email,
        recipient_name,
        subject,
        body,
        poster_path=None
    ):
        """
        Creates the complete email message.
        """

        message = EmailMessage()

        # ------------------------------------------------------
        # SENDER
        # ------------------------------------------------------

        message["From"] = formataddr(
            (
                self.sender_name,
                self.sender_email
            )
        )

        # ------------------------------------------------------
        # RECIPIENT
        # ------------------------------------------------------

        message["To"] = recipient_email

        # ------------------------------------------------------
        # SUBJECT
        # ------------------------------------------------------

        message["Subject"] = subject

        # ------------------------------------------------------
        # EMAIL BODY
        # ------------------------------------------------------

        message.set_content(
            body
        )

        # ------------------------------------------------------
        # POSTER ATTACHMENT
        # ------------------------------------------------------

        self._attach_poster(
            message=message,
            poster_path=poster_path
        )

        return message

    # ==========================================================
    # SEND ONE EMAIL
    # ==========================================================

    def send_email(
        self,
        recipient_email,
        recipient_name,
        subject,
        body,
        poster_path=None
    ):
        """
        Sends one email with the original poster attached.
        """

        # ------------------------------------------------------
        # CREATE MESSAGE
        # ------------------------------------------------------

        message = self._create_message(
            recipient_email=recipient_email,
            recipient_name=recipient_name,
            subject=subject,
            body=body,
            poster_path=poster_path
        )

        # ------------------------------------------------------
        # CONNECT TO SMTP SERVER
        # ------------------------------------------------------

        with smtplib.SMTP(
            self.smtp_server,
            self.smtp_port
        ) as server:

            # --------------------------------------------------
            # START TLS ENCRYPTION
            # --------------------------------------------------

            server.starttls()

            # --------------------------------------------------
            # LOGIN
            # --------------------------------------------------

            server.login(
                self.sender_email,
                self.sender_password
            )

            # --------------------------------------------------
            # SEND MESSAGE
            # --------------------------------------------------

            server.send_message(
                message
            )

        return True

    # ==========================================================
    # SEND MULTIPLE EMAILS
    # ==========================================================

    def send_emails(
        self,
        generated_emails
    ):
        """
        Sends emails to all filtered students.

        generated_emails should contain dictionaries such as:

        {
            "student_id": "ST001",
            "name": "Rahul Sharma",
            "email": "rahul@email.com",
            "subject": "...",
            "body": "...",
            "attachment_path": "poster.pdf"
        }
        """

        results = []

        for email_data in generated_emails:

            student_id = email_data.get(
                "student_id"
            )

            name = email_data.get(
                "name"
            )

            recipient_email = email_data.get(
                "email"
            )

            subject = email_data.get(
                "subject"
            )

            body = email_data.get(
                "body"
            )

            attachment_path = email_data.get(
                "attachment_path"
            )

            try:

                # --------------------------------------------------
                # VALIDATE EMAIL
                # --------------------------------------------------

                if not recipient_email:

                    raise ValueError(
                        "Recipient email is missing."
                    )

                if not body:

                    raise ValueError(
                        "Email body is empty."
                    )

                # --------------------------------------------------
                # SEND
                # --------------------------------------------------

                self.send_email(
                    recipient_email=recipient_email,
                    recipient_name=name,
                    subject=subject,
                    body=body,
                    poster_path=attachment_path
                )

                results.append(
                    {
                        "student_id": student_id,
                        "name": name,
                        "email": recipient_email,
                        "status": "Sent",
                        "error": ""
                    }
                )

            except Exception as error:

                results.append(
                    {
                        "student_id": student_id,
                        "name": name,
                        "email": recipient_email,
                        "status": "Failed",
                        "error": str(error)
                    }
                )

        return results