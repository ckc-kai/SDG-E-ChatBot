
from models.schemas import Source


class GenerationService:
    def __init__(self):
        # TODO :
        # from llm.generator import AnswerGenerator
        # self._generator = AnswerGenerator()
        pass

    def generate(
        self,
        question: str,
        sources: list[Source],
        model: str | None = None,
    ) -> str:
        # STUB: replace with a real call once code is ready 
        return f"This is a placeholder answer to: '{question}'"