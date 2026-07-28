"""Strict failures raised by authoritative brain computation and storage."""


class BrainComputeError(Exception):
    """Base class for failures that must not degrade to legacy brain state."""


class BrainValidationError(BrainComputeError):
    """An input or reconstructed state violates the declared brain domain."""


class BrainDurabilityError(BrainComputeError):
    """Authoritative state could not be validated or durably acknowledged."""


class BrainAllocationError(BrainDurabilityError):
    """A store allocation does not match the authoritative input state."""


class BrainCounterExhaustedError(BrainDurabilityError):
    """A durable monotone counter cannot advance without wrapping."""


class BrainOwnershipError(BrainComputeError):
    """A writer without the required brain capability attempted a mutation."""


class BrainNotificationBackpressureError(BrainComputeError):
    """A listener-context event could not reserve bounded notification capacity."""
