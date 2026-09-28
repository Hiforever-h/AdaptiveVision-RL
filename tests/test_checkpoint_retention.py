import tempfile
import unittest
from pathlib import Path
import json

from adaptive_vision_rl.verl.checkpoints import (
    install_checkpoint_retention,
    prune_local_checkpoints,
    validate_lora_adapter,
)


class CheckpointRetentionTests(unittest.TestCase):
    @staticmethod
    def make_checkpoint(root, step):
        actor = root / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        for name in ("model", "optim", "extra_state"):
            (actor / f"{name}_world_size_1_rank_0.pt").touch()
        (actor.parent / "data.pt").touch()

    def test_prunes_old_complete_and_partial_steps_after_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (20, 40, 60):
                self.make_checkpoint(root, step)
            (root / "global_step_10").mkdir()
            (root / "global_step_80").mkdir()
            (root / "unrelated").mkdir()
            (root / "latest_checkpointed_iteration.txt").write_text("60")

            removed = prune_local_checkpoints(root, keep=2)

            self.assertEqual(
                {path.name for path in removed},
                {"global_step_10", "global_step_20", "global_step_80"},
            )
            self.assertTrue((root / "global_step_40").is_dir())
            self.assertTrue((root / "global_step_60").is_dir())
            self.assertTrue((root / "unrelated").is_dir())

    def test_never_prunes_without_complete_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "global_step_20").mkdir()
            (root / "latest_checkpointed_iteration.txt").write_text("20")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                prune_local_checkpoints(root, keep=2)
            self.assertTrue((root / "global_step_20").is_dir())

    def test_one_checkpoint_is_removed_before_first_resumed_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, 20)
            (root / "latest_checkpointed_iteration.txt").write_text("20")
            old_exists_during_save = []

            class Trainer:
                def _load_checkpoint(self):
                    return None

                def _save_checkpoint(self):
                    old_exists_during_save.append((root / "global_step_20").exists())
                    CheckpointRetentionTests.make_checkpoint(root, 40)
                    (root / "latest_checkpointed_iteration.txt").write_text("40")

            trainer = Trainer()
            install_checkpoint_retention(trainer, root, keep=1)
            trainer._load_checkpoint()
            trainer._save_checkpoint()

            self.assertEqual(old_exists_during_save, [False])
            self.assertFalse((root / "global_step_20").exists())
            self.assertTrue((root / "global_step_40").is_dir())

    def test_two_checkpoint_retention_keeps_old_until_save_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, 20)
            (root / "latest_checkpointed_iteration.txt").write_text("20")
            old_exists_during_save = []

            class Trainer:
                def _load_checkpoint(self):
                    return None

                def _save_checkpoint(self):
                    old_exists_during_save.append((root / "global_step_20").exists())
                    CheckpointRetentionTests.make_checkpoint(root, 40)
                    (root / "latest_checkpointed_iteration.txt").write_text("40")

            trainer = Trainer()
            install_checkpoint_retention(trainer, root, keep=2)
            trainer._load_checkpoint()
            trainer._save_checkpoint()

            self.assertEqual(old_exists_during_save, [True])
            self.assertTrue((root / "global_step_20").is_dir())
            self.assertTrue((root / "global_step_40").is_dir())

    def test_rejects_empty_lora_safetensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "global_step_20" / "actor" / "lora_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}")
            empty_header = b"{}       "
            (adapter / "adapter_model.safetensors").write_bytes(
                len(empty_header).to_bytes(8, "little") + empty_header
            )
            self.assertEqual((adapter / "adapter_model.safetensors").stat().st_size, 17)
            with self.assertRaisesRegex(RuntimeError, "zero tensors"):
                validate_lora_adapter(root, 20)

    def test_accepts_lora_safetensors_with_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "global_step_20" / "actor" / "lora_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}")
            header = json.dumps(
                {"lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
            ).encode()
            (adapter / "adapter_model.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + b"\0" * 4
            )
            self.assertEqual(validate_lora_adapter(root, 20), 1)

    def test_save_reports_empty_adapter_even_when_pt_checkpoint_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class Trainer:
                global_steps = 20

                def _load_checkpoint(self):
                    return None

                def _save_checkpoint(self):
                    CheckpointRetentionTests.make_checkpoint(root, 20)
                    adapter = root / "global_step_20" / "actor" / "lora_adapter"
                    adapter.mkdir()
                    (adapter / "adapter_config.json").write_text("{}")
                    empty_header = b"{}       "
                    (adapter / "adapter_model.safetensors").write_bytes(
                        len(empty_header).to_bytes(8, "little") + empty_header
                    )
                    (root / "latest_checkpointed_iteration.txt").write_text("20")

            trainer = Trainer()
            install_checkpoint_retention(trainer, root, keep=1, expect_lora=True)
            with self.assertRaisesRegex(RuntimeError, "zero tensors"):
                trainer._save_checkpoint()
            self.assertTrue(
                (root / "global_step_20" / "actor" / "model_world_size_1_rank_0.pt").exists()
            )


if __name__ == "__main__":
    unittest.main()
