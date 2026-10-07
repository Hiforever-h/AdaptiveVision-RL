import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from adaptive_vision_rl.verl.consistency import _markdown_report
from adaptive_vision_rl.verl.short_run import install_short_run
from adaptive_vision_rl.verl.update_diagnostics import ratio_statistics, update_mode_log_probs
from scripts.run_dtpo_micro2_probe import package, short_config
from tests import test_forward_diagnostics as forward_fixture


class UpdateForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        forward_fixture.ForwardTraceTests.setUpClass()
        cls.model = forward_fixture.ForwardTraceTests.model
        cls.data = forward_fixture.ForwardTraceTests.data
        cls.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def actor(self, fail=False):
        def forward(micro_batch, temperature, calculate_entropy):
            self.assertFalse(calculate_entropy)
            mm = micro_batch["multi_modal_inputs"]
            output = self.model(
                **{key: micro_batch[key] for key in ("input_ids", "attention_mask")},
                position_ids=micro_batch["position_ids"].transpose(0, 1),
                **{key: torch.cat([entry[key] for entry in mm]) for key in mm[0]}, use_cache=False)
            if fail:
                raise RuntimeError("injected after checkpointed forward")
            width = micro_batch["responses"].shape[-1]
            logits = output.logits[:, -width-1:-1] / temperature
            return None, F.log_softmax(logits, -1).gather(-1, micro_batch["responses"][..., None]).squeeze(-1)
        return SimpleNamespace(actor_module=self.model, _forward_micro_batch=forward)

    def test_real_qwen_mixed_image_forward_enables_checkpointing_grad_and_restores_state(self):
        self.model.eval()
        # Also preserve intentionally mixed child modes, not just the root flag.
        self.model.model.visual.train()
        modes = [module.training for module in self.model.modules()]
        before = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        batch = {**self.data.batch, **self.data.non_tensor_batch}
        evidence = {}
        with torch.no_grad():
            actual = update_mode_log_probs(self.actor(), [batch, batch], temperature=1., evidence=evidence)
        self.assertTrue(evidence["valid"])
        self.assertTrue(evidence["training_modes_restored"])
        self.assertEqual(evidence["outputs_require_grad"], [True, True])
        self.assertEqual(actual.shape, (4, 5))
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual[:2], actual[2:], atol=0, rtol=0)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_forward_failure_restores_modes_and_removes_hooks(self):
        self.model.eval()
        evidence = {}
        with self.assertRaisesRegex(RuntimeError, "injected"):
            update_mode_log_probs(self.actor(fail=True), [{**self.data.batch, **self.data.non_tensor_batch}],
                                  temperature=1., evidence=evidence)
        self.assertFalse(self.model.training)
        self.assertTrue(evidence["training_modes_restored"])
        self.assertTrue(all(not module._forward_pre_hooks for module in self.model.modules()))

    def test_ratio_uses_log_difference_and_ignores_padding_and_nonfinite_tokens(self):
        # The tiny absolute probabilities must still expose ratios outside PPO bounds.
        old = torch.tensor([[-20., -20., -20., -20., float("nan")]])
        new = old + torch.tensor([[0., -.3, .3, 10., 0.]])
        result = ratio_statistics(old, new, torch.tensor([[1, 1, 1, 0, 1]], dtype=torch.bool))
        self.assertEqual(result["finite_tokens"], 3)
        self.assertEqual(result["nonfinite_tokens"], 1)
        self.assertAlmostEqual(result["fraction_outside_bounds"], 2/3)
        self.assertLess(result["min"], .8)
        self.assertGreater(result["max"], 1.24)

    def test_scoped_report_does_not_claim_current_vllm_or_optimizer_execution(self):
        text = _markdown_report(dict(status="completed", checkpoint="/ckpt", cases=[],
                                     update_forward_diagnostics=True))
        self.assertIn("不运行 vLLM prefill", text)
        self.assertIn("不调用 backward 或 optimizer", text)
        self.assertIn("它不是 pg_clipfrac", text)


class ShortTrainingTests(unittest.TestCase):
    def source(self):
        # Use the repository config shape without requiring OmegaConf/verl on CPU.
        return dict(actor_rollout_ref=dict(
            model=dict(path="/model/sft"), actor=dict(ppo_micro_batch_size_per_gpu=2,
                                                   optim=dict(total_training_steps=375)),
            rollout=dict(log_prob_micro_batch_size_per_gpu=4, multi_turn=dict(enable=False))),
            env=dict(adaptive_vision=dict(data_root="/data")),
            trainer=dict(total_training_steps=None, resume_mode="auto", resume_from_path="/ckpt300",
                         save_freq=20, test_freq=50, val_before_train=True, val_only=False))

    def trainer(self, config):
        class Node(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)
        def nodes(value):
            return Node(**{k: nodes(v) for k, v in value.items()}) if isinstance(value, dict) else value
        return SimpleNamespace(config=nodes(config), total_training_steps=375, train_dataloader=list(range(375)))

    def test_fresh_config_and_stop_at_10_preserve_full_optimizer_horizon_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = self.source()
            original = copy.deepcopy(source)
            config = short_config(source, output)
            self.assertEqual(source, original)
            self.assertEqual(config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"], 2)
            trainer = self.trainer(config)
            logs = []
            class Tracking:
                def log(self, data, step, backend=None):
                    logs.append(step)
            tracking = SimpleNamespace(Tracking=Tracking)
            recorder, restore = install_short_run(trainer, steps=10, output=output, tracking_module=tracking)
            try:
                self.assertEqual(trainer.total_training_steps, 10)
                self.assertEqual(trainer.config.actor_rollout_ref.actor.optim.total_training_steps, 375)
                logger = tracking.Tracking()
                for step in range(1, 11):
                    logger.log({"training/global_step": step, "actor/lr": 1e-6}, step=step)
                recorder.finish()
            finally:
                restore()
            self.assertIs(tracking.Tracking, Tracking)
            state = json.loads((output / "run_state.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["recorded_steps"], logs)
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 10)

    def test_incomplete_training_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = short_config(self.source(), output)
            tracking = SimpleNamespace(Tracking=type("Tracking", (), {}))
            recorder, restore = install_short_run(self.trainer(config), steps=10, output=output, tracking_module=tracking)
            try:
                recorder.record({"training/global_step": 1}, 1)
                with self.assertRaisesRegex(RuntimeError, "Expected training steps"):
                    recorder.finish()
                self.assertEqual(json.loads((output / "run_state.json").read_text())["status"], "failed")
            finally:
                restore()

    def test_accidental_resume_is_rejected(self):
        config = short_config(self.source(), Path("/tmp/probe"))
        config["trainer"]["resume_mode"] = "auto"
        with self.assertRaisesRegex(ValueError, "automatic resume disabled"):
            install_short_run(self.trainer(config), steps=10, output="/tmp/unused")

    def test_report_bundle_preserves_metrics_but_never_copies_weights(self):
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe"
            (output / "train").mkdir(parents=True)
            (output / "train/metrics.jsonl").write_text('{"step": 1}\n')
            (output / "model.pt").write_bytes(b"must not be packaged")
            (output / "adapter.safetensors").write_bytes(b"must not be packaged")
            with zipfile.ZipFile(package(output)) as archive:
                self.assertEqual(archive.namelist(), ["probe/train/metrics.jsonl"])
