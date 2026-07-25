from fastapi import Request

from videorag.api.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.videorag
