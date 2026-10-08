import copy
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest

import torch

from adaptive_vision_rl.answer_reward import encode_answer_reward
from adaptive_vision_rl.verl.reward_manager import AdaptiveVisionRewardManager
from adaptive_vision_rl.verl.validation_run import compare_validations, install_validation_run
from scripts.run_dtpo_val10_probe import build_command, exit_status, write_report
from tests.test_trajectory import FakeValidationData


class Node(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class ValidationRunTests(unittest.TestCase):
    def trainer(self):
        dtpo = Node(balance_penalty=.01, balance_threshold=.2, advantage_epsilon=1e-6, tool_advantage_coef=.3)
        config = Node(
            actor_rollout_ref=Node(
                actor=Node(ppo_micro_batch_size_per_gpu=2, ppo_epochs=1,
                           optim=Node(total_training_steps=375, lr_warmup_steps_ratio=.03)),
                rollout=Node(log_prob_micro_batch_size_per_gpu=2,
                             val_kwargs=Node(n=1, temperature=0., do_sample=False))),
            algorithm=Node(dtpo=dtpo),
            trainer=Node(resume_mode="disable", resume_from_path=None, save_freq=-1,
                         test_freq=10, val_before_train=True, val_only=False,
                         critic_warmup=0, total_training_steps=None))
        tokenizer = Node(decode=lambda tokens, **kwargs: " ".join(map(str, tokens)))
        trainer = Node(config=config, global_steps=0, total_training_steps=375,
                       train_dataloader=list(range(375)), val_dataset=[0, 1], tokenizer=tokenizer,
                       val_reward_fn=AdaptiveVisionRewardManager(tokenizer, dtpo), calls=[])

        def validate():
            trainer.calls.append(trainer.global_steps)
            data = FakeValidationData()
            data.batch = dict(responses=torch.tensor([[11, 0], [21, 0]]),
                              attention_mask=torch.tensor([[1, 1, 0], [1, 1, 0]]))
            if trainer.global_steps == 10:
                data.batch["responses"][1, 0] = 22
                data.non_tensor_batch["rewards"][1] = encode_answer_reward(
                    correct=True, score=1., format_reward=.1)
            result = trainer.val_reward_fn(data, return_dict=True)
            return {"val/success_rate": sum(result["reward_extra_info"]["dtpo_accuracy"])/2}

        trainer._validate = validate
        class Tracking:
            def log(self, data, step, backend=None):
                pass
        return trainer, Node(Tracking=Tracking)

    def test_validation_brackets_ten_updates_and_compares_stable_question_ids(self):
        trainer, tracking = self.trainer()
        original_validate, original_reward, original_tracking = trainer._validate, trainer.val_reward_fn, tracking.Tracking
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "train"
            recorder, restore = install_validation_run(trainer, steps=10, output=output, tracking_module=tracking)
            try:
                self.assertEqual(trainer.total_training_steps, 10)
                self.assertEqual(trainer.config.actor_rollout_ref.actor.optim.total_training_steps, 375)
                trainer._validate()
                logger = tracking.Tracking()
                # Match the pinned trainer: final validation occurs after the
                # last actor update, before that step's metrics are logged.
                for step in range(1, trainer.total_training_steps+1):
                    trainer.global_steps = step
                    if step == trainer.total_training_steps:
                        trainer._validate()
                    logger.log({"training/global_step": step, "actor/lr": 1e-6}, step)
                recorder.finish()
                state = json.loads((output / "run_state.json").read_text())
                comparison = json.loads((output / "validation_comparison.json").read_text())
                self.assertEqual(state["recorded_steps"], list(range(1, 11)))
                self.assertEqual(state["validation_steps"], [0, 10])
                self.assertEqual(state["status"], "completed")
                self.assertEqual(state["warmup_steps"], 11)
                self.assertEqual(trainer.calls, [0, 10])
                self.assertEqual(comparison["metrics"]["val/success_rate"], dict(before=.5, after=1., delta=.5))
                self.assertEqual(comparison["output_changed_count"], 1)
                self.assertEqual(comparison["wrong_to_correct_count"], 1)
                for phase in ("before", "after"):
                    samples = [json.loads(line) for line in (output / f"validation_{phase}_samples.jsonl").read_text().splitlines()]
                    self.assertEqual([row["sample_id"] for row in samples], ["question-a", "question-b"])
                    self.assertEqual(samples[0]["turns"][0]["token_ids"], [11])
                write_report(output.parent, {"status": "completed", "returncode": 0}, state)
                report = (output.parent / "REPORT.md").read_text()
                self.assertIn("val/success_rate | 0.5 | 1 | +0.5", report)
                self.assertIn("错误→正确 1 题", report)
            finally:
                restore()
            self.assertIs(trainer._validate, original_validate)
            self.assertIs(trainer.val_reward_fn, original_reward)
            self.assertIs(tracking.Tracking, original_tracking)

    def test_missing_final_validation_cannot_report_success(self):
        trainer, tracking = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            recorder, restore = install_validation_run(trainer, steps=10, output=directory, tracking_module=tracking)
            try:
                trainer._validate()
                for step in range(1, 11):
                    recorder.record({"training/global_step": step}, step)
                with self.assertRaisesRegex(RuntimeError, "both full validations"):
                    recorder.finish()
                recorder.fail("interrupted before final validation")
                self.assertEqual(json.loads((Path(directory) / "run_state.json").read_text())["status"], "failed")
                self.assertTrue((Path(directory) / "validation_before.json").exists())
            finally:
                restore()

    def test_extra_validation_and_incomplete_validation_are_rejected(self):
        trainer, tracking = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            recorder, restore = install_validation_run(trainer, steps=10, output=directory, tracking_module=tracking)
            try:
                trainer._validate()
                trainer.global_steps = 5
                with self.assertRaisesRegex(RuntimeError, "Unexpected validation"):
                    trainer._validate()
            finally:
                restore()
        trainer, tracking = self.trainer()
        trainer.val_dataset.append(2)
        with tempfile.TemporaryDirectory() as directory:
            recorder, restore = install_validation_run(trainer, steps=10, output=directory, tracking_module=tracking)
            try:
                with self.assertRaisesRegex(RuntimeError, "Incomplete validation"):
                    trainer._validate()
            finally:
                restore()

    def test_invalid_settings_are_rejected(self):
        for field, value, message in (("do_sample", True, "greedy validation"),
                                      ("n", 2, "greedy validation"), ("temperature", 1., "greedy validation")):
            trainer, tracking = self.trainer()
            setattr(trainer.config.actor_rollout_ref.rollout.val_kwargs, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                install_validation_run(trainer, steps=10, output="/tmp/unused-val10", tracking_module=tracking)
        trainer, tracking = self.trainer()
        trainer.config.actor_rollout_ref.actor.ppo_epochs = 2
        with self.assertRaisesRegex(ValueError, "one actor update"):
            install_validation_run(trainer, steps=10, output="/tmp/unused-val10", tracking_module=tracking)

    def test_unchanged_metrics_do_not_hide_changed_outputs_or_sample_mismatch(self):
        sample = dict(turns=[dict(output="answer A", token_ids=[1])], used_tool=False, accuracy=1., answer_score=1.)
        first = {"a": sample}
        last = copy.deepcopy(first)
        last["a"]["turns"] = [dict(output="answer B", token_ids=[2])]
        comparison = compare_validations({"val/accuracy": 1.}, {"val/accuracy": 1.}, first, last)
        self.assertFalse(comparison["any_metric_changed"])
        self.assertEqual(comparison["output_changed_count"], 1)
        with self.assertRaisesRegex(RuntimeError, "different sample ids"):
            compare_validations({"val/accuracy": 1.}, {"val/accuracy": 1.}, first, {"b": sample})
        with self.assertRaisesRegex(RuntimeError, "different metric keys"):
            compare_validations({"val/accuracy": 1.}, {"val/score": 1.}, first, first)


class ValidationEntrypointTests(unittest.TestCase):
    def test_command_enforces_fresh_val_train_val_and_keeps_schedule(self):
        output = Path("/tmp/val10")
        command = build_command(Path("/config.yaml"), output,
                                ["trainer.resume_mode=auto", "trainer.test_freq=2", "trainer.total_training_steps=375"])
        overrides = dict(item.split("=", 1) for item in command if "=" in item)
        self.assertEqual(overrides["trainer.resume_mode"], "disable")
        self.assertEqual(overrides["trainer.test_freq"], "10")
        self.assertEqual(overrides["trainer.total_training_steps"], "375")
        self.assertEqual(overrides["trainer.save_freq"], "-1")
        self.assertEqual(overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"], "2")
        self.assertEqual(overrides["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"], "2")
        self.assertIn("--probe-validation", command)
        self.assertNotIn("trainer.total_training_steps=10", command)

    def test_native_failure_cannot_be_masked_by_completed_results(self):
        training = dict(status="completed", recorded_steps=list(range(1, 11)), validation_steps=[0, 10])
        self.assertEqual(exit_status(0, training)["status"], "completed")
        for code in (-signal.SIGSEGV, 128+signal.SIGSEGV):
            with self.subTest(code=code):
                status = exit_status(code, training)
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["signal"], "SIGSEGV")
        self.assertEqual(exit_status(0, {})["status"], "failed")

    def test_failure_report_keeps_before_metrics_when_no_comparison_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "train").mkdir()
            (output / "train/validation_before.json").write_text('{"metrics":{"val/success_rate":0.5}}')
            write_report(output, dict(status="failed", returncode=1, error="injected"),
                         dict(recorded_steps=[1], validation_steps=[0], error="out of memory"))
            report = (output / "REPORT.md").read_text()
            self.assertIn("failed", report)
            self.assertIn("尚未完成", report)
            self.assertIn("out of memory", report)
            self.assertIn("injected", report)


if __name__ == "__main__":
    unittest.main()
