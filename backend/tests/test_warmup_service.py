import unittest
from unittest.mock import MagicMock, patch

from services.warmup_service import warm_application_models


class WarmupServiceTests(unittest.TestCase):
    @patch("services.warmup_service.get_reranker_model")
    @patch("services.warmup_service.get_embedding_model")
    def test_warms_all_local_model_boundaries(self, embedding, reranker):
        generation = MagicMock()

        with self.assertLogs("uvicorn.error", level="INFO"):
            report = warm_application_models(generation)

        embedding.assert_called_once_with()
        reranker.assert_called_once_with()
        generation.warmup.assert_called_once_with()
        self.assertTrue(report.ready)
        self.assertEqual(report.errors, ())

    @patch("services.warmup_service.get_reranker_model")
    @patch("services.warmup_service.get_embedding_model")
    def test_failure_is_recorded_without_aborting_startup(self, embedding, reranker):
        embedding.side_effect = RuntimeError("load failed")
        generation = MagicMock()

        with self.assertLogs("uvicorn.error", level="ERROR"):
            report = warm_application_models(generation)

        self.assertFalse(report.ready)
        self.assertEqual(report.errors, ("embedding",))
        reranker.assert_called_once_with()
        generation.warmup.assert_called_once_with()
