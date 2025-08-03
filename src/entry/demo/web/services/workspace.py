import re
import threading
from contextvars import ContextVar
from datetime import datetime, timedelta
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional

current_workspace: ContextVar[str] = ContextVar('current_workspace')
default_workspace: str = 'default'

_workspace_lock = threading.RLock()
_workspace_states: Dict[str, Dict[str, Any]] = {}

_workspace_states[default_workspace] = {
    'created_at': datetime.now(),
    'active': True,
    'data': {},
    'last_accessed': datetime.now()
}

LOGGER: Logger = getLogger(__name__)


class WorkspaceManager:
    """Thread-safe workspace management with proper validation and error handling."""

    @staticmethod
    def get_or_create_workspace(workspace_id: str) -> str:
        """Get existing workspace or create new one."""
        if not WorkspaceManager._is_valid_workspace_id(workspace_id):
            raise ValueError(f"Invalid workspace ID format: {workspace_id}")

        with _workspace_lock:
            if workspace_id not in _workspace_states:
                _workspace_states[workspace_id] = {
                    'created_at': datetime.now(),
                    'active': True,
                    'data': {},
                    'last_accessed': datetime.now()
                }
                LOGGER.info(f"Created workspace: {workspace_id}")
            else:
                # Update last accessed time
                _workspace_states[workspace_id]['last_accessed'] = datetime.now()

            return workspace_id

    @staticmethod
    def get_workspace_state(workspace_id: str) -> Dict[str, Any]:
        """Get workspace state, raise error if not found."""
        with _workspace_lock:
            if workspace_id not in _workspace_states:
                raise ValueError(f"Workspace '{workspace_id}' not found")

            _workspace_states[workspace_id]['last_accessed'] = datetime.now()
            return _workspace_states[workspace_id]

    @staticmethod
    def workspace_exists(workspace_id: str) -> bool:
        """Check if workspace exists."""
        with _workspace_lock:
            return workspace_id in _workspace_states

    @staticmethod
    def update_workspace_data(workspace_id: str, key: str, value: Any):
        """Update workspace data safely."""
        with _workspace_lock:
            if workspace_id not in _workspace_states:
                raise ValueError(f"Workspace '{workspace_id}' not found")

            _workspace_states[workspace_id]['data'][key] = value
            _workspace_states[workspace_id]['last_accessed'] = datetime.now()

    @staticmethod
    def get_workspace_data(workspace_id: str, key: str, default: Any = None) -> Any:
        """Get workspace data safely."""
        with _workspace_lock:
            if workspace_id not in _workspace_states:
                raise ValueError(f"Workspace '{workspace_id}' not found")

            _workspace_states[workspace_id]['last_accessed'] = datetime.now()
            return _workspace_states[workspace_id]['data'].get(key, default)

    @staticmethod
    def clear_workspace(workspace_id: str):
        """Clear workspace data but keep the workspace."""
        with _workspace_lock:
            if workspace_id not in _workspace_states:
                raise ValueError(f"Workspace '{workspace_id}' not found")

            _workspace_states[workspace_id]['data'] = {}
            _workspace_states[workspace_id]['last_accessed'] = datetime.now()
            LOGGER.info(f"Cleared workspace data: {workspace_id}")

    @staticmethod
    def delete_workspace(workspace_id: str):
        """Delete entire workspace."""
        if workspace_id == default_workspace:
            raise ValueError(f"Cannot delete default workspace: {workspace_id}")

        with _workspace_lock:
            if workspace_id not in _workspace_states:
                raise ValueError(f"Workspace '{workspace_id}' not found")

            del _workspace_states[workspace_id]
            LOGGER.info(f"Deleted workspace: {workspace_id}")

    @staticmethod
    def list_workspaces() -> List[str]:
        """List all workspace IDs."""
        with _workspace_lock:
            return list(_workspace_states.keys())

    @staticmethod
    def get_workspace_info() -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all workspaces."""
        with _workspace_lock:
            return {
                workspace_id: {
                    'created_at': state['created_at'].isoformat(),
                    'last_accessed': state['last_accessed'].isoformat(),
                    'active': state['active'],
                    'data_keys': list(state['data'].keys()),
                    'data_count': len(state['data'])
                }
                for workspace_id, state in _workspace_states.items()
            }

    @staticmethod
    def cleanup_inactive_workspaces(max_inactive_hours: int = 24):
        """Clean up workspaces that haven't been accessed recently."""
        cutoff_time = datetime.now() - timedelta(hours=max_inactive_hours)

        with _workspace_lock:
            inactive_workspaces = [
                workspace_id for workspace_id, state in _workspace_states.items()
                if (workspace_id != default_workspace and
                    state['last_accessed'] < cutoff_time)
            ]

            for workspace_id in inactive_workspaces:
                del _workspace_states[workspace_id]
                LOGGER.info(f"Cleaned up inactive workspace: {workspace_id}")

            return inactive_workspaces

    @staticmethod
    def _is_valid_workspace_id(workspace_id: str) -> bool:
        """Validate workspace ID format."""
        if not workspace_id or not isinstance(workspace_id, str):
            return False

        if len(workspace_id) < 1 or len(workspace_id) > 50:
            return False

        # Check format: alphanumeric, hyphens, underscores only
        if not re.match(r'^[a-zA-Z0-9_-]+$', workspace_id):
            return False

        # Reserved names
        reserved_names = {'admin', 'api', 'docs', 'health', 'metrics', 'static', 'root'}
        if workspace_id.lower() in reserved_names:
            return False

        return True


def get_current_workspace() -> str:
    """Get the current workspace from context."""
    try:
        return current_workspace.get()
    except LookupError:
        LOGGER.warning("No workspace in context, returning default")
        return default_workspace


def set_current_workspace(workspace_id: str):
    """Set the current workspace in context."""
    current_workspace.set(workspace_id)
    LOGGER.debug(f"Set current workspace to: {workspace_id}")


def get_workspace_dependency() -> str:
    """FastAPI dependency to get current workspace."""
    return get_current_workspace()


def with_workspace_context(workspace_id: str, func, *args, **kwargs):
    """Execute function with workspace context set."""
    old_workspace = current_workspace.get(None)
    try:
        set_current_workspace(workspace_id)
        return func(*args, **kwargs)
    finally:
        if old_workspace:
            set_current_workspace(old_workspace)


__all__ = [
    'WorkspaceManager',
    'get_current_workspace',
    'set_current_workspace',
    'default_workspace'
]