import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from services.warmup_service import WarmupReport


class MainEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_reports_model_state(self):
        app.state.model_warmup = WarmupReport(ready=True)
        response = self.client.get("/api/health")
        self.assertEqual(
            response.json(),
            {"status": "ok", "models_ready": True},
        )

    @patch("main.warm_application_models")
    @patch("main.ask.get_generation_service")
    def test_browser_warmup_refreshes_model_residency(self, get_service, warm):
        service = get_service.return_value
        warm.return_value = WarmupReport(ready=True, total_ms=12)

        response = self.client.post("/api/warmup")

        self.assertEqual(
            response.json(),
            {"status": "ready", "models_ready": True},
        )
        warm.assert_called_once_with(service)
        self.assertTrue(app.state.model_warmup.ready)


if __name__ == "__main__":
    unittest.main()
