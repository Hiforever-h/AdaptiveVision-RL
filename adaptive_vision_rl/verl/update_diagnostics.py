"""Forward-only checks of the actor path used by PPO updates."""

from __future__ import annotations

import torch


def ratio_statistics(old, new, mask, *, clip_low=.2, clip_high=.24):
    """Measure exp(new-old), before any update, on valid response tokens only.

    Outside-bound ratios are not pg_clipfrac: PPO clipping also depends on the
    sign of the advantage, which this fixed-response probe does not calculate.
    """
    old, new, mask = old.detach().double().cpu(), new.detach().double().cpu(), mask.bool().cpu()
    if old.shape != new.shape or old.shape != mask.shape:
        raise ValueError("old/new logprobs and response mask must have identical shapes")
    ratios = (new - old).exp()
    valid = mask & torch.isfinite(old) & torch.isfinite(new) & torch.isfinite(ratios)
    result = dict(requested_tokens=int(mask.sum()), finite_tokens=int(valid.sum()),
                  nonfinite_tokens=int((mask & ~valid).sum()), lower_bound=1 - clip_low,
                  upper_bound=1 + clip_high)
    if valid.any():
        selected = ratios[valid]
        below, above = selected < 1 - clip_low, selected > 1 + clip_high
        result.update(mean=float(selected.mean()), min=float(selected.min()), max=float(selected.max()),
                      p01=float(selected.quantile(.01)), p50=float(selected.quantile(.5)),
                      p99=float(selected.quantile(.99)),
                      fraction_below=float(below.double().mean()), fraction_above=float(above.double().mean()),
                      fraction_outside_bounds=float((below | above).double().mean()),
                      mean_abs_log_ratio=float((new[valid] - old[valid]).abs().mean()))
    return result


def update_mode_log_probs(actor, micro_batches, *, temperature, evidence):
    """Call the actual actor micro-forward with train/grad/checkpointing active.

    The caller supplies the same multimodal micro-batch dictionaries as
    update_policy. No loss, backward, gradient zeroing or optimizer is invoked.
    Detaching each output immediately releases its forward graph.
    """
    model = actor.actor_module
    modes = [(module, module.training) for module in model.modules()]
    checkpoint_layers = [module for module in model.modules()
                         if type(module).__name__ == "Qwen3VLTextDecoderLayer"]
    evidence.update(micro_batch_size=2, calculate_entropy=False, backward_called=False,
                    optimizer_called=False, forward_calls=[], checkpoint_layer_calls=[],
                    gradient_checkpointing_enabled=bool(getattr(model, "is_gradient_checkpointing", False)))
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Forward-only probe requires an actor without accumulated gradients")
    if not checkpoint_layers or not evidence["gradient_checkpointing_enabled"]:
        raise ValueError("Real update-mode verification requires Qwen3-VL gradient checkpointing")

    def model_hook(module, _):
        evidence["forward_calls"].append(dict(training=module.training, grad_enabled=torch.is_grad_enabled()))

    def checkpoint_hook(module, _):
        evidence["checkpoint_layer_calls"].append(dict(
            training=module.training, grad_enabled=torch.is_grad_enabled(),
            gradient_checkpointing=bool(module.gradient_checkpointing)))

    hooks = [model.register_forward_pre_hook(model_hook),
             checkpoint_layers[0].register_forward_pre_hook(checkpoint_hook)]
    outputs = []
    try:
        model.train()
        with torch.enable_grad():
            for micro_batch in micro_batches:
                if len(micro_batch["input_ids"]) != 2:
                    raise ValueError("Update forward must use exactly two rows per micro-batch")
                entropy, log_probs = actor._forward_micro_batch(
                    micro_batch=micro_batch, temperature=temperature, calculate_entropy=False)
                evidence.setdefault("outputs_require_grad", []).append(log_probs.requires_grad)
                outputs.append(log_probs.detach().float().cpu())
                del log_probs, entropy
        evidence["parameter_gradients_created"] = any(p.grad is not None for p in model.parameters())
        calls, layers = evidence["forward_calls"], evidence["checkpoint_layer_calls"]
        evidence["valid"] = (bool(outputs) and len(calls) == len(outputs) and len(layers) == len(outputs)
                             and all(item["training"] and item["grad_enabled"] for item in calls)
                             and all(all(item.values()) for item in layers)
                             and all(evidence["outputs_require_grad"])
                             and not evidence["parameter_gradients_created"])
        if not evidence["valid"]:
            raise RuntimeError("Update-mode forward evidence is incomplete; see report.json")
        return torch.cat(outputs)
    finally:
        for hook in hooks:
            hook.remove()
        for module, training in modes:
            module.training = training
        evidence["training_modes_restored"] = all(module.training == training for module, training in modes)


def update_comparisons(values, mask, *, clip_low, clip_high):
    from adaptive_vision_rl.verl.consistency import probability_difference

    pairs = [("actor_old_micro4", "actor_old_micro2"),
             ("actor_old_micro4", "actor_update_micro2"),
             ("actor_old_micro2", "actor_update_micro2"),
             ("actor_update_micro2", "actor_update_micro2_repeat")]
    probabilities, ratios = {}, {}
    for old, new in pairs:
        if old in values and new in values:
            name = f"{old}_vs_{new}"
            probabilities[name] = probability_difference(values[old], values[new], mask)
            ratios[name] = ratio_statistics(values[old], values[new], mask, clip_low=clip_low, clip_high=clip_high)
    for phase in ("actor_old_micro4", "actor_old_micro2"):
        if phase in values:
            probabilities[f"rollout_vs_{phase}"] = probability_difference(values["rollout"], values[phase], mask)
    return probabilities, ratios
