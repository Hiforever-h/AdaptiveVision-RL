"""Actor-side hook loaded by verl-agent's ``model.external_lib`` mechanism."""

from adaptive_vision_rl.verl.dtpo import dtpo_policy_loss


def _install():
    import verl.workers.actor.dp_actor as dp_actor

    dp_actor.compute_policy_loss = dtpo_policy_loss


_install()
