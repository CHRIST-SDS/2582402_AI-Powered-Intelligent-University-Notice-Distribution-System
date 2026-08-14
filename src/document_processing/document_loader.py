import os
from pathlib import Path
from typing import List

import easyocr
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


class PosterDocumentLoader:
    """
    Handles document loading and OCR preprocessing for the
    AI-Powered Smart University Notice Distribution System.

    Supported formats:
        - PDF
        - DOCX
        - TXT
        - PNG
        - JPG
        - JPEG
        - BMP
        - TIFF
        - WEBP
    """

    SUPPORTED_IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".webp",
    }

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        *SUPPORTED_IMAGE_EXTENSIONS,
    }

    def __init__(self, languages=None, gpu=False):
        """
        Initialize the EasyOCR reader.

        Parameters
        ----------
        languages : list, optional
            Languages supported by EasyOCR.
            Defaults to English.
        gpu : bool
            Whether to use GPU acceleration.
        """

        if languages is None:
            languages = ["en"]

        self.reader = easyocr.Reader(
            languages,
            gpu=gpu
        )

        print("EasyOCR Ready!")

    # ---------------------------------------------------------
    # OCR
    # ---------------------------------------------------------

    def perform_ocr(self, image_path: str) -> str:
        """
        Extract text from an image using EasyOCR.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        str
            Extracted text.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        result = self.reader.readtext(
            str(image_path),
            detail=0
        )

        extracted_text = "\n".join(result).strip()

        return extracted_text

    # ---------------------------------------------------------
    # PDF
    # ---------------------------------------------------------

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF document using PyPDFLoader.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        loader = PyPDFLoader(
            str(file_path)
        )

        documents = loader.load()

        print(
            f"Loaded {len(documents)} PDF pages "
            f"from {file_path.name}"
        )

        return documents

    # ---------------------------------------------------------
    # DOCX
    # ---------------------------------------------------------

    def load_docx(self, file_path: str) -> List[Document]:
        """
        Load a DOCX document using Docx2txtLoader.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"DOCX file not found: {file_path}"
            )

        loader = Docx2txtLoader(
            str(file_path)
        )

        documents = loader.load()

        print(
            f"Loaded {len(documents)} DOCX document(s) "
            f"from {file_path.name}"
        )

        return documents

    # ---------------------------------------------------------
    # TXT
    # ---------------------------------------------------------

    def load_txt(self, file_path: str) -> List[Document]:
        """
        Load a TXT document using TextLoader.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"TXT file not found: {file_path}"
            )

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        documents = loader.load()

        print(
            f"Loaded {len(documents)} TXT document(s) "
            f"from {file_path.name}"
        )

        return documents

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------

    def load_image(self, file_path: str) -> List[Document]:
        """
        Extract text from an image using OCR and return it
        as a LangChain Document.

        Note:
        The original image path is preserved in metadata so
        the same image can later be passed to the VLM.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {file_path}"
            )

        extracted_text = self.perform_ocr(
            str(file_path)
        )

        document = Document(
            page_content=extracted_text,
            metadata={
                "source": file_path.name,
                "source_path": str(file_path),
                "type": "image_notice",
                "ocr": True,
            }
        )

        print(
            f"OCR completed for: {file_path.name}"
        )

        return [document]

    # ---------------------------------------------------------
    # SINGLE DOCUMENT
    # ---------------------------------------------------------

    def load_single_document(
        self,
        file_path: str
    ) -> List[Document]:
        """
        Automatically detect the file type and load
        a single document.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension == ".pdf":

            return self.load_pdf(
                str(file_path)
            )

        elif extension == ".docx":

            return self.load_docx(
                str(file_path)
            )

        elif extension == ".txt":

            return self.load_txt(
                str(file_path)
            )

        elif extension in self.SUPPORTED_IMAGE_EXTENSIONS:

            return self.load_image(
                str(file_path)
            )

        else:

            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported formats: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

    # ---------------------------------------------------------
    # LOAD ALL DOCUMENTS
    # ---------------------------------------------------------

    def load_all_documents(
        self,
        input_directory: str
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        """

        input_directory = Path(input_directory)

        if not input_directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {input_directory}"
            )

        all_documents = []

        for file_path in input_directory.iterdir():

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                print(
                    f"Skipped unsupported file: "
                    f"{file_path.name}"
                )
                continue

            try:

                documents = self.load_single_document(
                    str(file_path)
                )

                all_documents.extend(
                    documents
                )

            except Exception as error:

                print(
                    f"Error processing "
                    f"{file_path.name}: {error}"
                )

        print(
            f"\nTotal Documents Loaded: "
            f"{len(all_documents)}"
        )

        return all_documents


if __name__ == "__main__":

    # Example usage
    loader = PosterDocumentLoader()

    sample_file = (
        "data/input/sample_poster.png"
    )

    if os.path.exists(sample_file):

        documents = loader.load_single_document(
            sample_file
        )

        print("\n========== OCR OUTPUT ==========\n")

        print(
            documents[0].page_content
        )

    else:

        print(
            f"Sample file not found: {sample_file}"
        )