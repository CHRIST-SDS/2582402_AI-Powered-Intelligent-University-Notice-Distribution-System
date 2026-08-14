import ollama

from src.vector_store.vector_store import PosterVectorStore


class PosterRAG:
    """
    RAG pipeline for answering questions about the
    currently uploaded poster.

    The pipeline retrieves information only from the
    current poster stored in Chroma and uses Gemma 3
    to generate the final answer.
    """

    def __init__(
        self,
        vector_store=None,
        model_name="gemma3:latest"
    ):
        self.vector_store = vector_store or PosterVectorStore()
        self.model_name = model_name

    def retrieve_context(self, question: str):
        """
        Retrieve the most relevant information from
        the currently stored poster.
        """

        results = self.vector_store.search(
            query=question,
            n_results=1
        )

        if not results["documents"]:
            return None

        documents = results["documents"][0]

        if not documents:
            return None

        return documents[0]

    def generate_answer(
        self,
        question: str,
        context: str
    ):
        """
        Generate an answer using Gemma 3 based only
        on the retrieved poster context.
        """

        prompt = f"""
You are a poster information assistant.

Answer the user's question using ONLY the information
provided in the poster context below.

Do not use outside knowledge.

Do not invent or assume information.

If the answer cannot be determined from the poster,
say:

"That information is not available on the uploaded poster."

Keep the answer concise and directly answer the question.

POSTER CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

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

    def ask(self, question: str):
        """
        Complete RAG pipeline:
        Question → Retrieval → Context → Gemma 3 → Answer
        """

        context = self.retrieve_context(question)

        if context is None:
            return (
                "That information is not available "
                "on the uploaded poster."
            )

        answer = self.generate_answer(
            question=question,
            context=context
        )

        return answer