import os
from logging import Logger, getLogger
from typing import Annotated, Optional

from fastapi import Header, HTTPException, Query
from starlette.status import HTTP_400_BAD_REQUEST

LOGGER: Logger = getLogger(__name__)

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