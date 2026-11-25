"""Offline checks for DISTRIBUTED_SEQUENTIAL_ALL_WORKERS (one notebook, every worker)."""

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from src.distributed_execution import configure_worker_paths
# Shared import helper: patch.dict(sys.modules) here would drop src modules
# imported inside it and leave later tests with duplicate module copies.
from test_deferred_solve_resume import orchestration


class SequentialWorkersTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.config = {
            "DISTRIBUTED_EXECUTION_ENABLED": True,
            "DISTRIBUTED_SEQUENTIAL_ALL_WORKERS": True,
            "DISTRIBUTED_RUN_ID": "seq-run",
            "DISTRIBUTED_WORKER_ID": 2,
            "DISTRIBUTED_WORKER_COUNT": 4,
            "BASE_OUTPUT_DIR": self.temporary.name,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def run_loop(self, worker_states, remote_states=None, stop_after=None):
        """Run the loop with a fake single-worker run that writes a local shard."""
        ran, roots_seen = [], []
        remote_states = remote_states or {}

        def fake_single_worker(experiment_configs, config, *args):
            self.assertTrue(config["_DISTRIBUTED_SEQUENTIAL_ACTIVE"])
            worker_id = config["DISTRIBUTED_WORKER_ID"]
            paths = configure_worker_paths(config)
            Path(config["RESULTS_DIR"], "run_log.json").write_text("[]")
            config["_DISTRIBUTED_MANIFEST"] = {"big": "manifest"}
            roots_seen.append(paths["worker_root"])
            ran.append(worker_id)
            if stop_after is not None and worker_id == stop_after:
                config["_DISTRIBUTED_STOP_NEW_BATCH_MONOTONIC"] = time.monotonic() - 1
            config["_DISTRIBUTED_LAST_WORKER_STATE"] = worker_states[worker_id]
            return {"exp": [{"query_index": 0}]}

        def fake_remote_state(config):
            return remote_states.get(config["DISTRIBUTED_WORKER_ID"])

        with patch.object(orchestration, "run_experiments", side_effect=fake_single_worker), \
                patch.object(orchestration, "remote_distributed_worker_state", side_effect=fake_remote_state):
            result = orchestration._run_distributed_workers_sequentially(
                [{"experiment_name": "exp"}], self.config, ["q"], None, {}, {}, ["a"],
            )
        return result, ran, roots_seen

    def test_runs_every_worker_in_order_and_frees_each_one(self):
        result, ran, roots = self.run_loop({i: "COMPLETE" for i in range(4)})
        self.assertEqual(ran, [0, 1, 2, 3])
        self.assertEqual(result, {})
        for root in roots:
            self.assertFalse(Path(root).exists(), root)
        self.assertEqual(self.config["DISTRIBUTED_WORKER_ID"], 2)
        self.assertNotIn("_DISTRIBUTED_SEQUENTIAL_ACTIVE", self.config)
        self.assertNotIn("_DISTRIBUTED_MANIFEST", self.config)
        self.assertEqual(
            self.config["_DISTRIBUTED_SEQUENTIAL_WORKER_STATES"],
            {0: "COMPLETE", 1: "COMPLETE", 2: "COMPLETE", 3: "COMPLETE"},
        )

    def test_skips_workers_already_complete_on_the_hub(self):
        _, ran, _ = self.run_loop(
            {i: "COMPLETE" for i in range(4)},
            remote_states={0: "COMPLETE", 1: "INCOMPLETE", 2: "COMPLETE"},
        )
        self.assertEqual(ran, [1, 3])

    def test_incomplete_worker_does_not_stop_the_loop(self):
        _, ran, _ = self.run_loop({0: "COMPLETE", 1: "INCOMPLETE", 2: "COMPLETE", 3: "COMPLETE"})
        self.assertEqual(ran, [0, 1, 2, 3])
        self.assertEqual(self.config["_DISTRIBUTED_SEQUENTIAL_WORKER_STATES"][1], "INCOMPLETE")

    def test_time_limit_stops_before_the_next_worker(self):
        _, ran, _ = self.run_loop({0: "COMPLETE", 1: "PAUSED_TIME_LIMIT"})
        self.assertEqual(ran, [0, 1])
        _, ran, _ = self.run_loop({0: "COMPLETE", 1: "COMPLETE"}, stop_after=0)
        self.assertEqual(ran, [0])

    def test_error_keeps_local_files_and_clears_the_loop_marker(self):
        def failing(experiment_configs, config, *args):
            configure_worker_paths(config)
            Path(config["RESULTS_DIR"], "unsynced.json").write_text("{}")
            raise RuntimeError("sync failed")

        with patch.object(orchestration, "run_experiments", side_effect=failing), \
                patch.object(orchestration, "remote_distributed_worker_state", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                orchestration._run_distributed_workers_sequentially(
                    [{"experiment_name": "exp"}], self.config, ["q"], None, {}, {}, ["a"],
                )
        self.assertTrue(Path(self.config["RESULTS_DIR"], "unsynced.json").exists())
        self.assertNotIn("_DISTRIBUTED_SEQUENTIAL_ACTIVE", self.config)

    def test_run_experiments_dispatches_only_when_enabled(self):
        def call(config):
            with patch.object(orchestration, "_check_api_routing"), \
                    patch.object(orchestration, "_require_standard_pipeline_managers"), \
                    patch.object(orchestration, "_check_api_batching"), \
                    patch.object(orchestration, "_check_exemplar_corpus"), \
                    patch.object(orchestration, "_run_distributed_workers_sequentially",
                                 return_value={"looped": []}) as loop, \
                    patch.object(orchestration, "_prepare_distributed_worker",
                                 side_effect=RuntimeError("single worker")):
                try:
                    return orchestration.run_experiments(
                        [{"experiment_name": "exp"}], config, ["q"], None, {}, {}, ["a"],
                    ), loop
                except RuntimeError as error:
                    return str(error), loop

        result, loop = call(self.config)
        self.assertEqual(result, {"looped": []})
        loop.assert_called_once()
        for disabled in (
            {**self.config, "DISTRIBUTED_SEQUENTIAL_ALL_WORKERS": False},
            {**self.config, "_DISTRIBUTED_SEQUENTIAL_ACTIVE": True},
        ):
            result, loop = call(disabled)
            self.assertEqual(result, "single worker")
            loop.assert_not_called()

    def test_flag_is_off_by_default_and_outside_the_manifest(self):
        import config as project_config
        from src.distributed_execution import sanitize_scientific_config

        self.assertFalse(project_config.CONFIG["DISTRIBUTED_SEQUENTIAL_ALL_WORKERS"])
        self.assertNotIn(
            "DISTRIBUTED_SEQUENTIAL_ALL_WORKERS",
            sanitize_scientific_config({"DISTRIBUTED_SEQUENTIAL_ALL_WORKERS": True, "X": 1}),
        )


class RemoteWorkerStateTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "DISTRIBUTED_EXECUTION_ENABLED": True,
            "DISTRIBUTED_RUN_ID": "seq-run",
            "DISTRIBUTED_WORKER_ID": 1,
            "DISTRIBUTED_WORKER_COUNT": 4,
            "PERSIST_RESULTS_ONLINE": True,
            "HF_SYNC_TOKEN": "token",
            "HF_HUB_USERNAME": "user",
            "HF_HUB_REPO_NAME": "repo",
        }

    def state_for(self, payload_text=None, error=None):
        from src import hf_sync

        def fake_download(**kwargs):
            self.assertEqual(
                kwargs["filename"], "distributed_runs/seq-run/workers/worker-001/status.json"
            )
            if error is not None:
                raise error
            path = Path(kwargs["local_dir"], "status.json")
            path.write_text(payload_text, encoding="utf-8")
            return str(path)

        with patch.object(hf_sync, "hf_hub_download", side_effect=fake_download):
            return hf_sync.remote_distributed_worker_state(self.config)

    def test_reads_matching_status(self):
        payload = '{"run_id": "seq-run", "worker_id": 1, "worker_count": 4, "state": "complete"}'
        self.assertEqual(self.state_for(payload), "COMPLETE")

    def test_foreign_or_missing_status_is_none(self):
        foreign = '{"run_id": "other", "worker_id": 1, "worker_count": 4, "state": "COMPLETE"}'
        self.assertIsNone(self.state_for(foreign))
        self.assertIsNone(self.state_for(error=OSError("network down")))


if __name__ == "__main__":
    unittest.main()
