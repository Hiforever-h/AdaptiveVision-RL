"""Record full validation on both sides of ten real, fresh DTPO updates."""

from __future__ import annotations

import json
import time

from .short_run import ShortRunRecorder, _json_value, install_short_run


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def compare_validations(before, after, before_samples, after_samples):
    """Align by stable question id, never by the randomly generated trajectory id."""
    if set(before_samples) != set(after_samples):
        raise RuntimeError("Before/after validation contains different sample ids")
    if set(before) != set(after):
        raise RuntimeError("Before/after validation contains different metric keys")
    metrics = {}
    for key in sorted(before):
        first, last = before[key], after[key]
        if (isinstance(first, bool) or isinstance(last, bool)
                or not isinstance(first, (int, float)) or not isinstance(last, (int, float))):
            raise RuntimeError(f"Nonfinite or nonnumeric validation metric: {key}")
        metrics[key] = dict(before=first, after=last, delta=last-first)
    changes = []
    for sample_id, first in before_samples.items():
        last = after_samples[sample_id]
        changes.append(dict(
            sample_id=sample_id,
            output_changed=first["turns"] != last["turns"],
            tool_use_changed=first["used_tool"] != last["used_tool"],
            accuracy_before=first["accuracy"], accuracy_after=last["accuracy"],
            accuracy_delta=last["accuracy"]-first["accuracy"],
            answer_score_before=first["answer_score"], answer_score_after=last["answer_score"],
        ))
    return dict(
        before_step=0, after_step=10, sample_count=len(changes), metrics=metrics,
        any_metric_changed=any(item["delta"] != 0 for item in metrics.values()),
        output_changed_count=sum(item["output_changed"] for item in changes),
        tool_use_changed_count=sum(item["tool_use_changed"] for item in changes),
        wrong_to_correct_count=sum(item["accuracy_delta"] > 0 for item in changes),
        correct_to_wrong_count=sum(item["accuracy_delta"] < 0 for item in changes),
        samples=changes,
    )


class ValidationRunRecorder(ShortRunRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state.update(validation=True, validation_steps=[], phase="initializing")
        self.validations = {}
        self.samples = {}
        self.phase = None
        self.persist()

    def begin_validation(self, step, expected_samples):
        completed = self.state["validation_steps"]
        expected_step = 0 if not completed else self.state["requested_steps"]
        if len(completed) >= 2 or step != expected_step:
            raise RuntimeError(f"Unexpected validation at step {step}; completed: {completed}")
        expected_training = list(range(1, step))
        if self.state["recorded_steps"] != expected_training:
            raise RuntimeError("Validation occurred before the expected training updates")
        self.phase = "before" if step == 0 else "after"
        self.samples[self.phase] = {}
        self.expected_samples = expected_samples
        self.validation_started = time.monotonic()
        (self.output / f"validation_{self.phase}_samples.jsonl").touch(exist_ok=False)
        self.state.update(phase="validation_" + self.phase, active_validation_step=step)
        self.persist()
        print(f"[val10] Starting full validation {self.phase}, step={step}, samples={expected_samples}", flush=True)

    def capture_samples(self, data, tokenizer, dtpo_config):
        from .trajectory import extract_trajectory_views

        if self.phase is None:
            raise RuntimeError("Validation reward was called outside a recorded validation")
        views = extract_trajectory_views(data, dtpo_config)
        responses = data.batch["responses"]
        masks = data.batch["attention_mask"][:, -responses.shape[-1]:]
        records = []
        for view in views:
            turns = []
            for row in view.row_indices:
                tokens = responses[row][masks[row].bool()].detach().cpu().tolist()
                turns.append(dict(stage="tool" if row == view.tool_row else "answer",
                                  output=tokenizer.decode(tokens, skip_special_tokens=True), token_ids=tokens))
            reward = view.reward
            sample_id = str(reward.group_id)
            record = dict(sample_id=sample_id, turns=turns, used_tool=bool(reward.used_tool),
                          accuracy=float(reward.accuracy), answer_score=float(reward.answer_score),
                          outcome_reward=float(reward.outcome_reward), tool_reward=float(reward.tool_reward),
                          vision_tokens_acquired=view.vision_tokens_acquired,
                          vision_tokens_processed=view.vision_tokens_processed)
            if sample_id in self.samples[self.phase]:
                raise RuntimeError(f"Duplicate validation sample id: {sample_id}")
            self.samples[self.phase][sample_id] = record
            records.append(record)
        with (self.output / f"validation_{self.phase}_samples.jsonl").open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def end_validation(self, metrics, step):
        if len(self.samples[self.phase]) != self.expected_samples:
            raise RuntimeError(f"Incomplete validation: expected {self.expected_samples} samples, "
                               f"received {len(self.samples[self.phase])}")
        clean = _json_value(metrics)
        if not clean or any(value is None for value in clean.values()):
            raise RuntimeError("Validation returned empty or nonfinite metrics")
        result = dict(step=step, metrics=clean, sample_count=self.expected_samples,
                      elapsed_seconds=time.monotonic()-self.validation_started)
        _write_json(self.output / f"validation_{self.phase}.json", result)
        self.validations[self.phase] = clean
        self.state["validation_steps"].append(step)
        self.state.update(phase="training" if step == 0 else "finishing", active_validation_step=None)
        self.persist()
        print(f"[val10] Completed full validation {self.phase}, step={step}", flush=True)
        self.phase = None

    def finish(self):
        if self.state["validation_steps"] != [0, self.state["requested_steps"]]:
            raise RuntimeError("Expected both full validations at steps 0 and 10")
        comparison = compare_validations(self.validations["before"], self.validations["after"],
                                         self.samples["before"], self.samples["after"])
        _write_json(self.output / "validation_comparison.json", comparison)
        self.state["phase"] = "finished"
        super().finish()


def install_validation_run(trainer, *, steps, output, tracking_module=None):
    """Use the pinned trainer's initial/final validation without changing its loop."""
    config = trainer.config
    val = config.actor_rollout_ref.rollout.val_kwargs
    if val.do_sample or int(val.n) != 1 or float(val.temperature) != 0:
        raise ValueError("Validation probe requires greedy validation, n=1 and temperature=0")
    if int(config.actor_rollout_ref.actor.ppo_epochs) != 1 or config.trainer.critic_warmup > 1:
        raise ValueError("Validation probe requires one actor update per step from step 1")
    if trainer.val_reward_fn is None or len(trainer.val_dataset) < 1:
        raise ValueError("A nonempty validation dataset and reward function are required")
    recorder, restore_tracking = install_short_run(
        trainer, steps=steps, output=output, tracking_module=tracking_module,
        validation=True, recorder_class=ValidationRunRecorder)
    original_validate, original_reward = trainer._validate, trainer.val_reward_fn

    def recorded_reward(data, return_dict=False):
        result = original_reward(data, return_dict=return_dict)
        recorder.capture_samples(data, trainer.tokenizer, config.algorithm.dtpo)
        return result

    def recorded_validate():
        step = int(trainer.global_steps)
        recorder.begin_validation(step, len(trainer.val_dataset))
        result = original_validate()
        recorder.end_validation(result, step)
        return result

    trainer.val_reward_fn = recorded_reward
    trainer._validate = recorded_validate

    def restore():
        trainer.val_reward_fn, trainer._validate = original_reward, original_validate
        restore_tracking()

    return recorder, restore
