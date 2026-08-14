from src.document_processing.document_loader import PosterDocumentLoader


loader = PosterDocumentLoader()

poster_path = r"C:\Python_project\Generative AI\Christ_University_Posters\christ_poster_1.jpg"

documents = loader.load_single_document(poster_path)

print("\n========== OCR OUTPUT ==========\n")

print(documents[0].page_content)