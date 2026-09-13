"""Lazy skeletal-transform facade; no model assets or animation state."""


def forward_kinematics(local, parents):
    from .torch.skeleton import forward_kinematics as operation
    return operation(local, parents)


__all__ = ["forward_kinematics"]
