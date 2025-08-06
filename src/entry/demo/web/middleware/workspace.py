import os
from logging import Logger, getLogger
from typing import Annotated, Optional, Final

from fastapi import Header, HTTPException, Query
from starlette.status import HTTP_400_BAD_REQUEST

from entry.demo.web.config import settings
from utils import get_folder_size

LOGGER: Logger = getLogger(__name__)
MAX_DISK_SPACE: Final[int] = 5 * 1024 * 1024 * 1024

def get_workspace(
        x_workspace: Annotated[Optional[str], Header()] = None,
        workspace: Optional[str] = Query(None)
) -> str:
    """
    Extract the workspace from either header or query parameter.
    Priority: Header > Query Parameter > Default
    """
    workspace_id = x_workspace or workspace

    if not workspace_id:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Workspace identifier is required. Provide via X-Workspace header or workspace query parameter."
        )

    if not workspace_id.isalnum():
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Workspace identifier must be alphanumeric"
        )

    LOGGER.info(f"Using workspace: {workspace_id}")
    return workspace_id


def get_workspace_path(workspace_id: str) -> str:
    """Generate workspace-specific file path"""
    base_path = os.getenv("WORKSPACE_BASE_PATH", "/app/workspaces")
    workspace_path = os.path.join(base_path, workspace_id)

    os.makedirs(workspace_path, exist_ok=True)

    return workspace_path

def validate_disk_space() -> bool:
    current_disk_size = get_folder_size(settings.DATA_DIR)
    if current_disk_size > settings.MAX_SERVER_DATA_SIZE:
        return False
    return True
