import importlib.util
import sys
import unittest
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

from adaptive_vision_rl.verl.lora_sync import (
    _patch_initial_lora_sync,
    collect_lora_params,
    install_lora_sync_fixes,
)


HAS_PEFT = all(importlib.util.find_spec(name) is not None for name in ("torch", "peft"))


@unittest.skipUnless(HAS_PEFT, "requires torch and peft")
class LoraSyncTests(unittest.TestCase):
    def setUp(self):
        import torch
        from peft import LoraConfig, get_peft_model

        self.torch = torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(2, 2, bias=False)
                self.k_proj = torch.nn.Linear(2, 2, bias=False)

            def forward(self, x):
                return self.q_proj(x)

        self.actor = get_peft_model(
            Model(), LoraConfig(r=1, lora_alpha=2, target_modules=["q_proj", "k_proj"])
        )
        with torch.no_grad():
            for layer in (self.actor.base_model.model.q_proj, self.actor.base_model.model.k_proj):
                layer.base_layer.weight.fill_(1)
                layer.lora_A.default.weight.fill_(1)
                layer.lora_B.default.weight.fill_(0.5)

        class Manager:
            def __init__(manager):
                manager.module = self.actor
                manager.base_sync_done = False
                manager.events = []
                manager.adapter = None

            def update_params(manager, params, peft_config=None):
                if peft_config is not None and manager.base_sync_done:
                    manager.events.append("adapter")
                    manager.adapter = params
                else:
                    manager.events.append("base")
                    manager.base = params
                    manager.base_sync_done = True

        self.manager_cls = Manager
        _patch_initial_lora_sync(Manager)
        self.manager = Manager()
        self.config = self.actor.peft_config["default"]

    def sync(self):
        if self.manager.base_sync_done:
            params = collect_lora_params(self.actor)
        else:
            params = {"q_proj.weight": self.actor.base_model.model.q_proj.base_layer.weight.detach().clone()}
        self.manager.update_params(params, peft_config=self.config)

    def assert_rollout_matches_actor(self):
        torch = self.torch
        x = torch.ones(1, 2)
        result = x @ self.manager.base["q_proj.weight"].T
        if self.manager.adapter is not None:
            a = self.manager.adapter["base_model.model.q_proj.lora_A.weight"]
            b = self.manager.adapter["base_model.model.q_proj.lora_B.weight"]
            result += (x @ a.T @ b.T) * (self.config.lora_alpha / self.config.r)
        torch.testing.assert_close(result, self.actor(x))

    def test_first_resumed_rollout_and_subsequent_updates_match_actor(self):
        self.sync()
        self.assertEqual(self.manager.events, ["base", "adapter"])
        self.assert_rollout_matches_actor()
        with self.torch.no_grad():
            self.actor.base_model.model.q_proj.lora_B.default.weight.fill_(0.75)
        self.sync()
        self.assertEqual(self.manager.events, ["base", "adapter", "adapter"])
        self.assert_rollout_matches_actor()

    def test_fresh_zero_b_adapter_is_loaded_and_is_valid(self):
        with self.torch.no_grad():
            for layer in (self.actor.base_model.model.q_proj, self.actor.base_model.model.k_proj):
                layer.lora_B.default.weight.zero_()
        self.sync()
        self.assertEqual(self.manager.events, ["base", "adapter"])
        self.assert_rollout_matches_actor()

    def test_preloaded_base_only_needs_one_adapter_update(self):
        self.manager.base_sync_done = True
        self.sync()
        self.assertEqual(self.manager.events, ["adapter"])

    def test_full_finetuning_path_does_not_collect_an_adapter(self):
        with patch("adaptive_vision_rl.verl.lora_sync.collect_lora_params") as collect:
            self.manager.update_params({"weight": object()})
        collect.assert_not_called()
        self.assertEqual(self.manager.events, ["base"])

    def test_installing_twice_does_not_duplicate_adapter_registration(self):
        _patch_initial_lora_sync(self.manager_cls)
        self.sync()
        self.assertEqual(self.manager.events, ["base", "adapter"])

    def test_collection_failure_aborts_first_sync_and_retry_loads_adapter(self):
        with patch("adaptive_vision_rl.verl.lora_sync.collect_lora_params", side_effect=ValueError("Incomplete")):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                self.sync()
        self.assertIsNone(self.manager.adapter)
        self.sync()
        self.assertEqual(self.manager.events, ["base", "adapter"])
        self.assert_rollout_matches_actor()

    def test_collector_rejects_empty_and_missing_whole_modules(self):
        state = collect_lora_params(self.actor)
        partial = {name: value for name, value in state.items() if ".k_proj." in name}
        for invalid in ({}, partial):
            with self.subTest(keys=list(invalid)):
                with patch("peft.utils.save_and_load.get_peft_model_state_dict", return_value=invalid):
                    with self.assertRaisesRegex(ValueError, "Incomplete"):
                        collect_lora_params(self.actor)

    def test_cpu_tensor_copies_survive_fsdp_context_storage_changes(self):
        actor = self.actor
        torch = self.torch

        class FSDP:
            def __init__(self):
                self._fsdp_wrapped_module = actor

            @staticmethod
            @contextmanager
            def summon_full_params(module, *, writeback):
                yield
                with torch.no_grad():
                    actor.base_model.model.q_proj.lora_B.default.weight.zero_()

        with patch("torch.distributed.fsdp.FullyShardedDataParallel", FSDP):
            state = collect_lora_params(FSDP())
        torch.testing.assert_close(
            state["base_model.model.q_proj.lora_B.weight"], torch.full((2, 1), 0.5)
        )


class LoraHookInstallationTests(unittest.TestCase):
    def test_installs_collector_in_all_upstream_consumers(self):
        names = (
            "verl", "verl.utils", "verl.utils.fsdp_utils", "verl.workers",
            "verl.workers.fsdp_workers", "verl.workers.sharding_manager",
            "verl.workers.sharding_manager.fsdp_vllm",
        )
        modules = {name: ModuleType(name) for name in names}
        manager_cls = type("Manager", (), {"update_params": lambda *args, **kwargs: None})
        modules[names[-1]].FSDPVLLMShardingManager = manager_cls
        with patch.dict(sys.modules, modules):
            install_lora_sync_fixes()
            installed_update = manager_cls.update_params
            install_lora_sync_fixes()
        self.assertIs(manager_cls.update_params, installed_update)
        for name in (names[2], names[4], names[-1]):
            self.assertIs(modules[name].layered_summon_lora_params, collect_lora_params)


if __name__ == "__main__":
    unittest.main()
