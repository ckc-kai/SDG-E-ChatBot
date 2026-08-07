"""
Location: backend/tests/test_documents_router.py
Role: tests GET /api/documents. 
connect_db is mocked at the point it's
imported INTO routers.documents
"""
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app


class TestDocumentsRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _mock_conn_with_rows(self, rows):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        return mock_conn

    @patch("routers.documents.connect_db")
    def test_list_documents_returns_rows_from_db(self, mock_connect_db):
        mock_connect_db.return_value = self._mock_conn_with_rows(
            [(1, "2024_Base-WMP.pdf"), (2, "2023_Base-WMP.pdf")]
        )

        response = self.client.get("/api/documents")

        self.assertEqual(response.status_code, 200)
        docs = response.json()["documents"]
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["title"], "2024_Base-WMP.pdf")
        self.assertEqual(docs[0]["doc_id"], "1")

    @patch("routers.documents.connect_db")
    def test_list_documents_handles_empty_table(self, mock_connect_db):
        mock_connect_db.return_value = self._mock_conn_with_rows([])

        response = self.client.get("/api/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documents"], [])

    @patch("routers.documents.connect_db")
    def test_list_documents_closes_connection(self, mock_connect_db):
        mock_conn = self._mock_conn_with_rows([])
        mock_connect_db.return_value = mock_conn

        self.client.get("/api/documents")

        mock_conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()