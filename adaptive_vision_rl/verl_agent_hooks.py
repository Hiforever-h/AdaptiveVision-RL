"""Actor-side hook loaded by verl-agent's ``model.external_lib`` mechanism."""

from adaptive_vision_rl.verl.dtpo import dtpo_policy_loss
from adaptive_vision_rl.vllm_compat import install_qwen3vl_lora_mapping_backport


def _install():
    # Must run before verl-agent constructs the vLLM rollout model. This keeps
    # LoRA on the language stack instead of wrapping Qwen3-VL visual layers.
    install_qwen3vl_lora_mapping_backport()

    import verl.workers.actor.dp_actor as dp_actor

    dp_actor.compute_policy_loss = dtpo_policy_loss


_install()
