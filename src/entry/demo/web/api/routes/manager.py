from fastapi import APIRouter

from entry.demo.web.services.workspace import WorkspaceManager

router = APIRouter()

@router.get("/workspace")
def list_all_workspace():
    return {
        "workspaces": WorkspaceManager.list_workspaces()
    }
