"""Read and management routes for built-in and template agents (``/v1/agents``).

Built-in agents are the long-lived, shared agents the server provides
out of the box — the seeded ``claude-native-ui`` agent plus anything
registered at startup with ``omnigent server --agent`` or promoted
by administrators via ``POST /v1/agents/promote``. They are the
``session_id IS NULL`` rows in ``agent_store``; ``agent_store.list()``
filters to exactly these. Session-scoped agents (created via
multipart ``POST /v1/sessions``) belong to one conversation and are read
through ``GET /v1/sessions/{id}/agent``.

The Web UI's new-session picker calls this to discover bindable
built-ins, then creates a session with
``POST /v1/sessions {agent_id, host_id, workspace}``. See
``designs/BUILTIN_AGENTS.md``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from omnigent.db.utils import builtin_agent_id, generate_agent_id
from omnigent.entities import Agent
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.runtime.agent_cache import AgentCache
from omnigent.server.auth import AuthProvider
from omnigent.server.routes._auth_helpers import get_user_id
from omnigent.server.routes._auth_helpers import require_user as _require_user
from omnigent.server.schemas import AgentObject, MCPServerSummary, PaginatedList, SkillSummary
from omnigent.stores import AgentStore, ConversationStore
from omnigent.stores.artifact_store import ArtifactStore
from omnigent.stores.permission_store import PermissionStore

_logger = logging.getLogger(__name__)


class AgentPromoteRequest(BaseModel):
    """Request payload to promote a custom/session agent to a global template."""

    agent_id: str | None = None
    session_id: str | None = None
    name: str | None = None
    description: str | None = None


async def _require_admin(
    request: Request,
    auth_provider: AuthProvider | None,
    permission_store: PermissionStore | None,
) -> None:
    """Verify caller has admin privileges."""
    if permission_store is None:
        return
    user_id = get_user_id(request, auth_provider)
    if user_id is None:
        raise OmnigentError("Authentication required", code=ErrorCode.UNAUTHORIZED)
    is_admin = await asyncio.to_thread(permission_store.is_admin, user_id)
    if not is_admin:
        raise OmnigentError(
            "Admin privileges required to manage template agents",
            code=ErrorCode.FORBIDDEN,
        )


def _to_agent_object(agent: Agent, agent_cache: AgentCache) -> AgentObject:
    """
    Convert a runtime Agent entity to an API-layer AgentObject.

    Loads the spec from cache to populate ``mcp_servers``,
    ``skills``, and (when the stored row has none) the
    ``description``; on any load failure those fall back to empty /
    the stored value rather than failing the whole list — one
    unreadable bundle must not break discovery.

    :param agent: The runtime agent entity, e.g. the seeded
        ``claude-native-ui`` agent.
    :param agent_cache: Cache used to load the agent spec.
    :returns: An :class:`AgentObject` for the API response.
    """
    mcp_servers: list[MCPServerSummary] = []
    skills: list[SkillSummary] = []
    terminals: list[str] = []
    harness: str | None = None
    # Prefer the stored entity's description; fall back to the spec's
    # top-level description when the stored value is unset (single-file
    # YAML agents don't persist it at registration today). Lets the
    # new-session picker show a hover description without a migration.
    description: str | None = agent.description
    try:
        # Built-ins are operator-authored template agents
        # (session_id is None), so ${VAR} expansion against the server
        # env is allowed here; a tenant session-scoped agent would not
        # expand.
        loaded = agent_cache.load(
            agent.id, agent.bundle_location, expand_env=agent.session_id is None
        )
        if description is None:
            description = loaded.spec.description
        # Declared terminal names, in spec order (mirrors the
        # session-agent endpoint so both report it consistently).
        terminals = list(loaded.spec.terminals or {})
        # Bundled skills only — host-discovered skills are runner-owned
        # and unknowable here (no session, no runner). The new-session
        # composer uses this list for its "/" menu.
        skills = [SkillSummary(name=s.name, description=s.description) for s in loaded.spec.skills]
        mcp_servers = [
            MCPServerSummary(
                name=srv.name,
                transport=srv.transport,
                description=srv.description,
                url=srv.url,
                headers=dict.fromkeys(srv.headers, "[REDACTED]") if srv.headers else {},
                command=srv.command,
                args=srv.args,
            )
            for srv in loaded.spec.mcp_servers
        ]
        # Kind for the Add Agent picker (Codex vs Claude). Stays None
        # when the bundle can't be loaded (the except below).
        harness = loaded.spec.executor.harness_kind
    except Exception:  # noqa: BLE001 — spec load failure must not break the list
        _logger.debug(
            "Failed to load spec for agent %s; mcp_servers/skills will be empty",
            agent.id,
            exc_info=True,
        )
    return AgentObject(
        id=agent.id,
        name=agent.name,
        version=agent.version,
        description=description,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        harness=harness,
        mcp_servers=mcp_servers,
        mcp_servers_editable=False,
        skills=skills,
        terminals=terminals,
        # Seeded built-ins use a deterministic, name-derived id; an
        # operator/user-registered template (e.g. ``--agent``) uses a
        # random id. The picker protects the former from being shadowed
        # by a same-named ``omnigent run`` upload, but lets a newer
        # upload supersede the latter.
        builtin=agent.session_id is None and agent.id == builtin_agent_id(agent.name),
    )


def create_builtin_agents_router(
    agent_store: AgentStore,
    agent_cache: AgentCache,
    *,
    auth_provider: AuthProvider | None = None,
    artifact_store: ArtifactStore | None = None,
    permission_store: PermissionStore | None = None,
    conversation_store: ConversationStore | None = None,
) -> APIRouter:
    """Build the router for ``/v1/agents`` (built-in & template agents).

    Mounted with ``prefix="/v1"`` so the final path is ``/v1/agents``.

    :param agent_store: Store whose ``list()`` returns only built-in
        (``session_id IS NULL``) agents.
    :param agent_cache: Cache for loading specs (populates
        ``mcp_servers`` on each agent).
    :param auth_provider: Optional auth provider; when set, the caller
        must be authenticated.
    :param artifact_store: Optional artifact store for promoting agent bundles.
    :param permission_store: Optional permission store for admin check.
    :param conversation_store: Optional conversation store for resolving session agents.
    :returns: A FastAPI router exposing discovery and admin promotion endpoints.
    """
    router = APIRouter()

    @router.get("/agents")
    async def list_builtin_agents(
        request: Request,
        limit: int = Query(default=20, ge=1, le=1000),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
    ) -> PaginatedList:
        """List built-in agents with cursor-based pagination.

        Returns only built-in agents — ``agent_store.list()`` filters
        ``session_id IS NULL`` — so session-scoped agents never appear.

        :param request: The incoming FastAPI request (for auth).
        :param limit: Maximum number of agents to return (1-1000).
        :param after: Cursor — return agents after this id.
        :param before: Cursor — return agents before this id.
        :param order: Sort order, ``"asc"`` or ``"desc"``.
        :returns: A :class:`PaginatedList` of built-in agents.
        """
        _require_user(request, auth_provider)
        page = agent_store.list(limit=limit, after=after, before=before, order=order)
        return PaginatedList(
            data=[_to_agent_object(a, agent_cache) for a in page.data],
            first_id=page.first_id,
            last_id=page.last_id,
            has_more=page.has_more,
        )

    @router.post("/agents/promote")
    async def promote_agent_to_template(
        request: Request,
        body: AgentPromoteRequest,
    ) -> AgentObject:
        """Promote a custom/session-scoped agent to a global template agent (Admin only)."""
        await _require_admin(request, auth_provider, permission_store)

        if artifact_store is None:
            raise OmnigentError("Artifact store not configured", code=ErrorCode.INTERNAL_ERROR)

        source_agent_id = body.agent_id
        if not source_agent_id and body.session_id and conversation_store is not None:
            conv = await asyncio.to_thread(conversation_store.get_conversation, body.session_id)
            if conv is not None and conv.agent_id:
                source_agent_id = conv.agent_id

        if not source_agent_id:
            raise OmnigentError(
                "Either agent_id or session_id must be provided",
                code=ErrorCode.INVALID_INPUT,
            )

        source_agent = await asyncio.to_thread(agent_store.get, source_agent_id)
        if source_agent is None:
            raise OmnigentError(f"Agent {source_agent_id!r} not found", code=ErrorCode.NOT_FOUND)

        try:
            bundle_bytes = await asyncio.to_thread(
                artifact_store.get, source_agent.bundle_location
            )
        except Exception as exc:
            raise OmnigentError(
                f"Failed to read source agent bundle: {exc}",
                code=ErrorCode.INTERNAL_ERROR,
            ) from exc

        target_name = (body.name or source_agent.name).strip()
        target_description = (
            body.description if body.description is not None else source_agent.description
        )

        bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
        existing_template = await asyncio.to_thread(agent_store.get_by_name, target_name)

        if existing_template is not None:
            new_loc = f"{existing_template.id}/{bundle_hash}"
            await asyncio.to_thread(artifact_store.put, new_loc, bundle_bytes)
            updated_agent = await asyncio.to_thread(
                agent_store.update, existing_template.id, new_loc
            )
            if updated_agent is not None:
                agent_cache.replace(existing_template.id, new_loc, bundle_bytes, expand_env=True)
                return _to_agent_object(updated_agent, agent_cache)

        template_agent_id = generate_agent_id()
        bundle_key = f"{template_agent_id}/{bundle_hash}"
        await asyncio.to_thread(artifact_store.put, bundle_key, bundle_bytes)
        created_agent = await asyncio.to_thread(
            agent_store.create,
            template_agent_id,
            target_name,
            bundle_key,
            description=target_description,
        )
        agent_cache.evict(template_agent_id)
        return _to_agent_object(created_agent, agent_cache)

    @router.delete("/agents/{agent_id}")
    async def delete_template_agent(
        request: Request,
        agent_id: str,
    ) -> dict[str, Any]:
        """Delete a custom registered template agent (Admin only)."""
        await _require_admin(request, auth_provider, permission_store)

        agent = await asyncio.to_thread(agent_store.get, agent_id)
        if agent is None:
            raise OmnigentError(f"Agent {agent_id!r} not found", code=ErrorCode.NOT_FOUND)

        if agent.session_id is not None:
            raise OmnigentError(
                "Cannot delete session-scoped agent through template endpoint",
                code=ErrorCode.INVALID_INPUT,
            )

        # Protect core seeded built-ins from accidental deletion
        if agent.id == builtin_agent_id(agent.name):
            raise OmnigentError(
                f"Cannot delete core seeded built-in agent {agent.name!r}",
                code=ErrorCode.FORBIDDEN,
            )

        deleted = await asyncio.to_thread(agent_store.delete, agent_id)
        agent_cache.evict(agent_id)
        return {"deleted": deleted, "id": agent_id}

    return router
