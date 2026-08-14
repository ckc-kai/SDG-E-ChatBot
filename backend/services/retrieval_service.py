
import logging

from retrieval.query.pdf import retrieve
from retrieval.utils import connect_db

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self):
        self._conn = connect_db()

    def retrieve_ranked_results(self, question: str):
        return retrieve(question, self._conn)