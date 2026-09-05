import unittest
from unittest.mock import Mock, patch

from src import api_manager


class APIRetryTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ENABLE_API_RETRY": True,
            "MAX_API_RETRIES": 3,
            "RETRY_ALL_API_ERRORS": True,
            "API_RETRY_DELAY_SECONDS": 0,
        }
        for name in ("global_pause_manager", "tprint", "_format_api_log", "_get_api_caller_location"):
            patcher = patch.object(api_manager, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_call(self, attempt):
        return api_manager.execute_with_retry(self.config, "test", "model", "prompt", attempt)

    def test_all_error_types_retry_and_recover(self):
        errors = (api_manager.NON_RETRYABLE_ERROR_TYPES | api_manager.RETRYABLE_ERROR_TYPES | {"NewProviderError"}) - {"SessionDeadline"}
        for error_type in sorted(errors):
            with self.subTest(error_type=error_type):
                success = {"status": "SUCCESS", "text": "answer"}
                attempt = Mock(side_effect=[{"status": "ERROR", "error_type": error_type}, success])
                self.assertEqual(self.run_call(attempt), success)
                self.assertEqual(attempt.call_count, 2)

    def test_escaped_exception_retries(self):
        attempt = Mock(side_effect=[ValueError("request failure"), {"status": "SUCCESS", "text": "answer"}])
        self.assertEqual(self.run_call(attempt)["status"], "SUCCESS")
        self.assertEqual(attempt.call_count, 2)

    def test_attempt_limit(self):
        attempt = Mock(side_effect=lambda: {"status": "ERROR", "error_type": "InvalidArgument"})
        self.assertTrue(self.run_call(attempt)["retry_exhausted"])
        self.assertEqual(attempt.call_count, 3)

    def test_disabled_retries(self):
        self.config["ENABLE_API_RETRY"] = False
        attempt = Mock(return_value={"status": "ERROR", "error_type": "RateLimitError"})
        self.run_call(attempt)
        self.assertEqual(attempt.call_count, 1)

    def test_selective_retry_policy(self):
        self.config["RETRY_ALL_API_ERRORS"] = False
        attempt = Mock(return_value={"status": "ERROR", "error_type": "AuthenticationError"})
        self.run_call(attempt)
        self.assertEqual(attempt.call_count, 1)
        attempt = Mock(side_effect=[{"status": "ERROR", "error_type": "RateLimitError"}, {"status": "SUCCESS"}])
        self.assertEqual(self.run_call(attempt)["status"], "SUCCESS")
        self.assertEqual(attempt.call_count, 2)

    def test_session_deadline_stops_without_request(self):
        attempt = Mock()
        with patch.object(api_manager, "api_deadline_due", return_value=True):
            self.assertEqual(self.run_call(attempt)["error_type"], "SessionDeadline")
        attempt.assert_not_called()

    def test_session_deadline_response_is_not_retried(self):
        attempt = Mock(return_value={"status": "ERROR", "error_type": "SessionDeadline"})
        self.run_call(attempt)
        self.assertEqual(attempt.call_count, 1)

    def test_keyboard_interrupt_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_call(Mock(side_effect=KeyboardInterrupt))


if __name__ == "__main__":
    unittest.main()
