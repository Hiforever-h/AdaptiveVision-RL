import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from adaptive_vision_rl.verl.lora_checkpoint import select_lora_tensors


HAS_PEFT = all(importlib.util.find_spec(name) is not None for name in ("torch", "peft"))


@unittest.skipUnless(HAS_PEFT, "requires torch and peft")
class CheckpointAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Preload native libraries before patch.dict restores sys.modules.
        import peft
        import torch.distributed

        # Load the real project worker, replacing only the absent verl runtime.
        names = (
            "verl", "verl.single_controller", "verl.single_controller.base",
            "verl.single_controller.base.decorator", "verl.utils", "verl.utils.fsdp_utils",
            "verl.workers", "verl.workers.fsdp_workers",
        )
        modules = {name: ModuleType(name) for name in names}
        modules[names[3]].Dispatch = SimpleNamespace(ONE_TO_ALL="one_to_all")
        modules[names[3]].register = lambda **kwargs: lambda method: method
        modules[names[5]].load_fsdp_model_to_gpu = Mock()
        modules[names[5]].offload_fsdp_model_to_cpu = Mock()
        modules[names[7]].ActorRolloutRefWorker = type("ActorRolloutRefWorker", (), {})
        source = Path(__file__).resolve().parents[1] / "adaptive_vision_rl/verl/worker.py"
        spec = importlib.util.spec_from_file_location("adaptive_vision_rl.verl._test_worker", source)
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(cls.module)

    def setUp(self):
        import torch
        from peft import LoraConfig, get_peft_model

        self.torch = torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.q_proj.weight.fill_(1)

            def forward(self, x):
                return self.q_proj(x)

        self.model_cls = Model
        self.actor = get_peft_model(Model(), LoraConfig(r=1, lora_alpha=2, target_modules=["q_proj"]))
        with torch.no_grad():
            self.actor.base_model.model.q_proj.lora_A.default.weight.fill_(1)
            self.actor.base_model.model.q_proj.lora_B.default.weight.fill_(0.5)
        self.worker = object.__new__(self.module.DTPOActorRolloutRefWorker)
        self.worker._is_actor = True
        self.worker._is_lora = True
        self.worker._is_offload_param = True
        self.worker.rank = 0
        self.worker.actor_module = self.actor
        self.worker.actor_module_fsdp = self.actor
        self.worker.checkpoint_manager = SimpleNamespace(save_checkpoint=Mock(side_effect=self.save_training_state))

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        base_path = self.directory / "base"
        base_path.mkdir()
        (base_path / "config.json").write_text("{}")
        self.actor.peft_config["default"].base_model_name_or_path = str(base_path)
        self.stdout = self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.barrier = self.stack.enter_context(patch.object(self.module.dist, "barrier"))
        self.stack.enter_context(patch.object(self.module.dist, "get_world_size", return_value=1))
        self.gather = self.stack.enter_context(patch.object(
            self.module.dist, "all_gather_object", side_effect=lambda gathered, item: gathered.__setitem__(0, item)
        ))
        self.broadcast = self.stack.enter_context(patch.object(self.module.dist, "broadcast_object_list"))
        self.module.load_fsdp_model_to_gpu.reset_mock()
        self.module.offload_fsdp_model_to_cpu.reset_mock()

    def save_training_state(self, *, local_path, **kwargs):
        actor = Path(local_path)
        actor.mkdir(parents=True)
        self.torch.save(self.actor.state_dict(), actor / "model_world_size_1_rank_0.pt")
        for name in ("optim", "extra_state"):
            (actor / f"{name}_world_size_1_rank_0.pt").touch()

    def save(self, step=20):
        actor = self.directory / f"global_step_{step}" / "actor"
        self.worker.save_checkpoint(str(actor), global_step=step, max_ckpt_to_keep=1)
        return actor

    def test_saves_current_adapter_and_config_alongside_training_state(self):
        from peft import PeftModel
        from safetensors.torch import load_file
        from scripts.evaluate_dtpo import resolve_lora_adapter

        actor_path = self.save()
        adapter = actor_path / "lora_adapter"
        self.assertEqual(resolve_lora_adapter(actor_path.parent), adapter.resolve())
        config = json.loads((adapter / "adapter_config.json").read_text())
        self.assertEqual(config["r"], 1)
        self.assertEqual(config["lora_alpha"], 2)
        self.assertEqual(config["base_model_name_or_path"], str(self.directory / "base"))
        state = self.torch.load(actor_path / "model_world_size_1_rank_0.pt", weights_only=True)
        expected = select_lora_tensors(state, rank=1)
        actual = load_file(adapter / "adapter_model.safetensors")
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            self.torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        restored = PeftModel.from_pretrained(self.model_cls(), adapter)
        x = self.torch.ones(1, 2)
        self.torch.testing.assert_close(restored(x), self.actor(x), rtol=0, atol=0)
        self.worker.checkpoint_manager.save_checkpoint.assert_called_once_with(
            local_path=str(actor_path), hdfs_path=None, global_step=20, max_ckpt_to_keep=1
        )
        self.module.offload_fsdp_model_to_cpu.assert_called_once_with(self.actor)
        self.assertIn("Saved 2 LoRA tensors", self.stdout.getvalue())

    def assert_nonfatal_adapter_failure(self, failing_patch):
        with failing_patch:
            actor_path = self.save()
        self.assertTrue((actor_path / "model_world_size_1_rank_0.pt").is_file())
        self.assertFalse((actor_path / "lora_adapter").exists())
        self.assertEqual(list(actor_path.glob("dtpo-adapter-*")), [])
        self.assertIn("Warning: LoRA adapter export failed", self.stdout.getvalue())
        self.assertIn("Training will continue", self.stdout.getvalue())
        self.assertNotIn("Saved 2 LoRA tensors", self.stdout.getvalue())
        self.module.offload_fsdp_model_to_cpu.assert_called_once_with(self.actor)
        # A later training checkpoint must still export updated weights.
        with self.torch.no_grad():
            self.actor.base_model.model.q_proj.lora_B.default.weight.fill_(0.75)
        from safetensors.torch import load_file

        next_path = self.save(step=40)
        tensors = load_file(next_path / "lora_adapter/adapter_model.safetensors")
        self.torch.testing.assert_close(tensors["base_model.model.q_proj.lora_B.weight"], self.torch.full((2, 1), 0.75))

    def test_collection_failure_keeps_checkpoint_and_training_continues(self):
        self.assert_nonfatal_adapter_failure(patch.object(
            self.module, "collect_lora_params", side_effect=ValueError("Incomplete default LoRA state")
        ))

    def test_empty_adapter_is_rejected_without_interrupting_training(self):
        self.assert_nonfatal_adapter_failure(patch.object(self.module, "collect_lora_params", return_value={}))

    def test_weight_write_failure_cleans_partial_adapter_and_training_continues(self):
        def partial_write(tensors, filename):
            Path(filename).write_bytes(b"partial")
            raise OSError("No space left on device")

        self.assert_nonfatal_adapter_failure(patch("safetensors.torch.save_file", side_effect=partial_write))

    def test_config_write_failure_cleans_partial_adapter_and_training_continues(self):
        self.assert_nonfatal_adapter_failure(patch.object(
            self.actor.peft_config["default"], "save_pretrained", side_effect=OSError("config write failed")
        ))

    def test_full_checkpoint_failure_still_raises_and_restores_offloading(self):
        self.worker.checkpoint_manager.save_checkpoint.side_effect = OSError("checkpoint write failed")
        with patch.object(self.module, "collect_lora_params") as collect:
            with self.assertRaisesRegex(OSError, "checkpoint write failed"):
                self.save()
        collect.assert_not_called()
        self.module.offload_fsdp_model_to_cpu.assert_called_once_with(self.actor)

    def test_full_finetuning_does_not_export_adapter(self):
        self.worker._is_lora = False
        with patch.object(self.module, "collect_lora_params") as collect:
            actor_path = self.save()
        collect.assert_not_called()
        self.assertFalse((actor_path / "lora_adapter").exists())

    def test_peer_collection_failure_warns_and_skips_adapter_write(self):
        self.gather.side_effect = lambda failures, item: failures.__setitem__(slice(None), [None, "rank 1: collection failed"])
        with patch.object(self.module.dist, "get_world_size", return_value=2), \
                patch.object(self.module, "save_adapter") as save:
            self.save()
        save.assert_not_called()
        self.assertIn("rank 1: collection failed", self.stdout.getvalue())

    def test_only_rank_zero_writes_adapter_files(self):
        self.worker.rank = 1
        self.gather.side_effect = lambda failures, item: failures.__setitem__(slice(None), [None, None])
        with patch.object(self.module.dist, "get_world_size", return_value=2), \
                patch.object(self.module, "save_adapter") as save:
            self.save()
        save.assert_not_called()
        self.broadcast.assert_called_once_with([None], src=0)


if __name__ == "__main__":
    unittest.main()
