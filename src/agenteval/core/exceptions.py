class AgentEvalError(Exception):
    """Base exception for AgentEval."""


class TraceError(AgentEvalError):
    """Raised when tracing state is invalid."""


class StorageError(AgentEvalError):
    """Raised when persistence fails."""
