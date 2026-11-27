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

    def run_loop(self, worker_states, remote_states=None, stop_after=None, verify_results=None):
        """Run the loop with a fake single-worker run that writes a local shard."""
        ran, roots_seen = [], []
        remote_states = remote_states or {}
        self.verified, self.resynced = [], []
        verify_results = list(verify_results or [])

        def fake_verify(config):
            # Verification must run while the local copy still exists.
            self.assertTrue(Path(config["_DISTRIBUTED_WORKER_ROOT"]).is_dir())
            self.verified.append(config["DISTRIBUTED_WORKER_ID"])
            if verify_results and not verify_results.pop(0):
                raise orchestration.DistributedSyncError("remote differs")
            return 1

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
                patch.object(orchestration, "remote_distributed_worker_state", side_effect=fake_remote_state), \
                patch.object(orchestration, "verify_distributed_worker_uploaded", side_effect=fake_verify), \
                patch.object(orchestration, "sync_workspace_to_hub",
                             side_effect=lambda config: self.resynced.append(config["DISTRIBUTED_WORKER_ID"])):
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

    def test_worker_range_runs_only_first_to_last_inclusive(self):
        self.config["DISTRIBUTED_WORKER_COUNT"] = 10
        self.config["DISTRIBUTED_SEQUENTIAL_WORKER_RANGE"] = [5, 9]
        _, ran, _ = self.run_loop({i: "COMPLETE" for i in range(10)}, remote_states={6: "COMPLETE"})
        self.assertEqual(ran, [5, 7, 8, 9])
        self.assertEqual(sorted(self.config["_DISTRIBUTED_SEQUENTIAL_WORKER_STATES"]), [5, 6, 7, 8, 9])
        self.config["DISTRIBUTED_SEQUENTIAL_WORKER_RANGE"] = (3, 3)
        _, ran, _ = self.run_loop({3: "COMPLETE"})
        self.assertEqual(ran, [3])

    def test_invalid_worker_range_is_rejected_before_any_worker_runs(self):
        for bad in ([5], [9, 5], [-1, 2], [0, 4], ["0", "2"], [True, 2], 5, "0-3"):
            with self.subTest(bad=bad):
                self.config["DISTRIBUTED_SEQUENTIAL_WORKER_RANGE"] = bad
                with patch.object(orchestration, "run_experiments") as single, \
                        patch.object(orchestration, "remote_distributed_worker_state") as remote:
                    with self.assertRaisesRegex(
                        orchestration.DistributedExecutionError, "DISTRIBUTED_SEQUENTIAL_WORKER_RANGE"
                    ):
                        orchestration._run_distributed_workers_sequentially(
                            [{"experiment_name": "exp"}], self.config, ["q"], None, {}, {}, ["a"],
                        )
                    single.assert_not_called()
                    remote.assert_not_called()
                self.assertNotIn("_DISTRIBUTED_SEQUENTIAL_ACTIVE", self.config)

    def test_each_worker_is_verified_on_the_hub_before_deletion(self):
        _, ran, roots = self.run_loop({i: "COMPLETE" for i in range(4)})
        self.assertEqual(self.verified, ran)
        self.assertEqual(self.resynced, [])

    def test_mismatch_is_uploaded_again_then_rechecked(self):
        _, ran, roots = self.run_loop({i: "COMPLETE" for i in range(4)}, verify_results=[False, True])
        self.assertEqual(self.resynced, [0])
        self.assertEqual(self.verified, [0, 0, 1, 2, 3])
        self.assertFalse(Path(roots[0]).exists())

    def test_second_mismatch_stops_and_keeps_the_local_copy(self):
        with self.assertRaisesRegex(orchestration.DistributedSyncError, "remote differs"):
            self.run_loop({i: "COMPLETE" for i in range(4)}, verify_results=[False, False])
        self.assertTrue(Path(self.config["_DISTRIBUTED_WORKER_ROOT"]).is_dir())
        self.assertNotIn("_DISTRIBUTED_SEQUENTIAL_ACTIVE", self.config)

    def test_sequential_worker_run_keeps_no_result_logs(self):
        results = orchestration._DiscardedResults()
        results["exp"] = [{"big": "log"}] * 1000
        self.assertEqual(results, {})

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
        self.assertIsNone(project_config.CONFIG["DISTRIBUTED_SEQUENTIAL_WORKER_RANGE"])
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


class UploadVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.config = {
            "DISTRIBUTED_EXECUTION_ENABLED": True,
            "PERSIST_RESULTS_ONLINE": True,
            "DISTRIBUTED_RUN_ID": "seq-run",
            "DISTRIBUTED_WORKER_ID": 1,
            "DISTRIBUTED_WORKER_COUNT": 4,
            "HF_SYNC_TOKEN": "token",
            "HF_HUB_USERNAME": "user",
            "HF_HUB_REPO_NAME": "repo",
            "BASE_OUTPUT_DIR": self.temporary.name,
        }
        configure_worker_paths(self.config)
        root = Path(self.config["_DISTRIBUTED_WORKER_ROOT"])
        (root / "status.json").write_bytes(b'{"state": "COMPLETE"}')
        (root / "results" / "exp_run_log.json").write_bytes(b"[" + b"1," * 5000 + b"1]")
        (root / "results" / "batch_journals").mkdir()
        (root / "logs" / "not_uploaded.log").write_text("x")
        self.root = root

    def tearDown(self):
        self.temporary.cleanup()

    def remote_entries(self, *, corrupt=None, missing=None):
        import hashlib
        from huggingface_hub.hf_api import RepoFile, RepoFolder

        prefix = "distributed_runs/seq-run/workers/worker-001"
        entries = [RepoFolder(path=prefix + "/results", oid="tree")]
        for relative, lfs in (("status.json", False), ("results/exp_run_log.json", True)):
            if relative == missing:
                continue
            data = (self.root / relative).read_bytes()
            if relative == corrupt:
                data = data[:-1] + b"2"
            info = {"path": prefix + "/" + relative, "size": len(data),
                    "oid": hashlib.sha1(b"blob %d" % len(data) + bytes([0]) + data).hexdigest()}
            if lfs:
                info["lfs"] = {"size": len(data), "oid": hashlib.sha256(data).hexdigest(),
                               "pointerSize": 130}
            entries.append(RepoFile(**info))
        return entries

    def verify(self, entries):
        from src import hf_sync

        with patch.object(hf_sync.HfApi, "list_repo_tree", return_value=iter(entries)):
            return hf_sync.verify_distributed_worker_uploaded(self.config)

    def test_matching_upload_passes(self):
        self.assertEqual(self.verify(self.remote_entries()), 2)

    def test_missing_or_changed_file_fails(self):
        from src.hf_sync import DistributedSyncError

        for kwargs in ({"missing": "status.json"}, {"corrupt": "status.json"},
                       {"corrupt": "results/exp_run_log.json"}):
            with self.subTest(**kwargs), self.assertRaises(DistributedSyncError):
                self.verify(self.remote_entries(**kwargs))


class RestoreMoveTests(unittest.TestCase):
    def test_atomic_move_leaves_no_second_copy(self):
        from src.hf_sync import _atomic_move

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary, "staging", "run_log.json")
            source.parent.mkdir()
            source.write_text("[1, 2]")
            destination = Path(temporary, "worker", "results", "run_log.json")
            _atomic_move(source, destination)
            self.assertFalse(source.exists())
            self.assertEqual(destination.read_text(), "[1, 2]")
            self.assertEqual(list(destination.parent.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
