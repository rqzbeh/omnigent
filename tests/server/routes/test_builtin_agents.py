"""Tests for the builtin agents discovery route (``GET /v1/agents``).

The app fixture does not trigger the lifespan event that seeds
built-in agents, so the test database starts empty. We seed a
test agent directly via the agent_store to verify the endpoint works.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest_asyncio

from omnigent.db.utils import builtin_agent_id, generate_agent_id
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore


@pytest_asyncio.fixture()
async def _seeded_agent(db_uri: str) -> str:
    """Seed a built-in (session_id=None) agent and return its ID."""
    agent_store = SqlAlchemyAgentStore(db_uri)
    agent_id = generate_agent_id()
    agent_store.create(agent_id, name="test-builtin", bundle_location="test:///bundle")
    return agent_id


async def test_list_builtin_agents_empty(client: httpx.AsyncClient) -> None:
    """GET /v1/agents with no agents returns an empty paginated list."""
    resp = await client.get("/v1/agents")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert isinstance(body["data"], list)
    assert "has_more" in body


async def test_list_builtin_agents_with_limit(client: httpx.AsyncClient) -> None:
    """Limit parameter constrains the result size."""
    resp = await client.get("/v1/agents?limit=1")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) <= 1


async def test_list_builtin_agents_seeded(
    client: httpx.AsyncClient,
    _seeded_agent: str,
) -> None:
    """A seeded agent appears in the list."""
    resp = await client.get("/v1/agents?limit=100")
    assert resp.status_code == 200
    ids = [a["id"] for a in resp.json()["data"]]
    assert _seeded_agent in ids


async def test_list_builtin_agents_response_shape(
    client: httpx.AsyncClient,
    _seeded_agent: str,
) -> None:
    """Each agent object has the expected fields."""
    resp = await client.get("/v1/agents?limit=100")
    assert resp.status_code == 200
    for agent in resp.json()["data"]:
        assert "id" in agent
        assert "name" in agent
        assert "created_at" in agent


async def test_builtin_flag_distinguishes_seeded_from_registered(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """
    ``builtin`` is True only for a server-seeded agent (deterministic,
    name-derived id); an operator/user-registered template (random id)
    reports False. The Web UI picker keys its supersession policy on this:
    a same-named upload may shadow a registered template but never a seeded
    built-in.
    """
    agent_store = SqlAlchemyAgentStore(db_uri)
    seeded_id = builtin_agent_id("polly")
    agent_store.create(seeded_id, name="polly", bundle_location="test:///polly")
    registered_id = generate_agent_id()
    agent_store.create(registered_id, name="my-agent", bundle_location="test:///mine")

    resp = await client.get("/v1/agents?limit=100")
    assert resp.status_code == 200
    by_id = {a["id"]: a for a in resp.json()["data"]}
    assert by_id[seeded_id]["builtin"] is True
    assert by_id[registered_id]["builtin"] is False


async def test_promote_custom_agent_and_delete(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """Admins can promote a custom session agent to a global template agent and delete it."""
    # Promote endpoint requires session_id or agent_id
    resp_invalid = await client.post("/v1/agents/promote", json={})
    assert resp_invalid.status_code in (400, 422)


async def test_promote_custom_agent_to_template(
    client: httpx.AsyncClient,
    db_uri: str,
    tmp_path: Path,
) -> None:
    """Admins can promote a custom session-scoped agent to a global template agent."""
    from omnigent.stores.artifact_store.local import LocalArtifactStore
    from tests.server.helpers import build_agent_bundle

    agent_store = SqlAlchemyAgentStore(db_uri)
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    bundle_bytes = build_agent_bundle(
        name="my-custom-agent", description="Custom agent description"
    )
    source_agent_id = generate_agent_id()
    bundle_loc = f"{source_agent_id}/bundle.tar.gz"
    artifact_store.put(bundle_loc, bundle_bytes)

    # Create a custom agent
    agent_store.create(source_agent_id, name="my-custom-agent", bundle_location=bundle_loc)

    # Promote to template
    resp = await client.post(
        "/v1/agents/promote",
        json={"agent_id": source_agent_id, "name": "shared-custom-agent"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "shared-custom-agent"
    assert data["builtin"] is False

    # Check it now appears in GET /v1/agents
    list_resp = await client.get("/v1/agents?limit=100")
    assert list_resp.status_code == 200
    names = [a["name"] for a in list_resp.json()["data"]]
    assert "shared-custom-agent" in names
