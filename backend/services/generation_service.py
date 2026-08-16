
from generation import AnswerRequest, AnswerService, adapt_ranked_results, create_provider_from_env


class GenerationService:
    def __init__(self):
        self._answer_service = AnswerService(create_provider_from_env())

    def warmup(self) -> None:
        self._answer_service.provider.warmup()

    def answer(self, request_id: str, question: str, ranked_results) -> dict:
        chunks = adapt_ranked_results(ranked_results)
        request = AnswerRequest(
            request_id=request_id,
            question=question,
            chunks=chunks,
        )
        response = self._answer_service.answer(request)
        return response.to_public_dict()