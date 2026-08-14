import os
import ollama


class LocalVisionModel:
    def __init__(self, model_name="gemma3"):
        self.model_name = model_name

    def extract_content(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        prompt = """
Look carefully at this university poster.

Your task is to READ the image and transcribe the information you can see.

Do not summarize yet.

Read the poster from top to bottom and report the actual visible text.

Pay special attention to:

- University name
- Department
- Event title
- Event type
- Dates
- Important dates
- Venue
- Email addresses
- Website URLs
- Registration information
- Contact information
- Eligibility
- Text near the QR code

If some text is genuinely unreadable, say "Unreadable" only for that specific item.

Do NOT invent information.

First describe what you can see on the poster, then provide the extracted text.

"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ]
        )

        return response["message"]["content"]