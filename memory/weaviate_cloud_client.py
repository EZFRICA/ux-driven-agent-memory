"""
Weaviate Cloud Client — Async Operations Layer.
All methods are now async to support high-concurrency with FastAPI.
"""

from datetime import datetime, timezone
import asyncio
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.classes.tenants import Tenant
from weaviate.util import generate_uuid5
from memory.schema import get_weaviate_client_async, init_all_schemas
from logger import get_logger

logger = get_logger(__name__)


# ── Tenant Management — Helper ────────────────────────────────────────────────

async def _ensure_tenant_async(collection, tenant_name: str) -> None:
    """Lazily create tenant if it doesn't exist (Async)."""
    try:
        # Weaviate collections in MT mode require explicit tenant creation
        await collection.tenants.create(tenants=[Tenant(name=tenant_name)])
        logger.debug("Tenant '%s' created for collection '%s'.", tenant_name, collection.name)
    except Exception as e:
        # Weaviate's Python client doesn't always have a clean 'TenantAlreadyExists' exception class
        # so we check the error message string for common success/conflict patterns.
        err_msg = str(e).lower()
        if "already exists" in err_msg or "409" in err_msg:
            return # Normal case: tenant is already there
        
        # Critical error (connection, auth, etc.)
        logger.error("Failed to ensure tenant '%s': %s", tenant_name, e)
        raise e


# ── BlockIndex operations (DLL routing) ────────────────────────────────────────

async def upsert_block_index(client, block_id: str, keywords: list[str], block_type: str, agent_id: str) -> None:
    """
    Insert or update a block's keyword vector in the BlockIndex collection (Async).
    Uses Tenant isolation.
    """
    collection = client.collections.get("BlockIndex")
    await _ensure_tenant_async(collection, agent_id)
    
    tenant_coll = collection.with_tenant(agent_id)
    keywords_text = " ".join(keywords)
    obj_uuid = generate_uuid5(f"{agent_id}_{block_id}")

    properties = {
        "keywords_text": keywords_text,
        "block_id": block_id,
        "agent_id": agent_id,
        "block_type": block_type,
        "is_active": True,
    }

    try:
        # Async CRUD
        await tenant_coll.data.update(uuid=obj_uuid, properties=properties)
        logger.debug("BlockIndex updated: '%s' for tenant '%s'.", block_id, agent_id)
    except Exception:
        # Not found → insert
        await tenant_coll.data.insert(uuid=obj_uuid, properties=properties)
        logger.debug("BlockIndex inserted: '%s' for tenant '%s'.", block_id, agent_id)


async def search_block_index(client, query: str, agent_id: str, limit: int = 12) -> list[dict]:
    """
    Semantic search on BlockIndex using near_text (Async).
    Returns: [{block_id, block_type, certainty}, ...]
    """
    collection = client.collections.get("BlockIndex")
    tenant_coll = collection.with_tenant(agent_id)
    
    response = await tenant_coll.query.near_text(
        query=query,
        limit=limit,
        target_vector="keywords_text",
        return_metadata=MetadataQuery(certainty=True),
    )

    results = []
    for obj in response.objects:
        results.append({
            "block_id": obj.properties.get("block_id"),
            "block_type": obj.properties.get("block_type"),
            "certainty": obj.metadata.certainty or 0.0,
        })

    return results


async def delete_block_index(client, block_id: str, agent_id: str) -> None:
    """Remove a block from the BlockIndex collection (Async)."""
    collection = client.collections.get("BlockIndex")
    tenant_coll = collection.with_tenant(agent_id)
    obj_uuid = generate_uuid5(f"{agent_id}_{block_id}")
    try:
        await tenant_coll.data.delete_by_id(obj_uuid)
        logger.debug("BlockIndex deleted: '%s' for tenant '%s'.", block_id, agent_id)
    except Exception as e:
        logger.warning("BlockIndex delete failed for '%s': %s", block_id, e)


async def fetch_all_block_indexes(client, agent_id: str) -> list[dict]:
    """Fetch all blocks from the BlockIndex for a specific tenant (Async)."""
    collection = client.collections.get("BlockIndex")
    tenant_coll = collection.with_tenant(agent_id)
    results = []
    try:
        response = await tenant_coll.query.fetch_objects(limit=100)
        for obj in response.objects:
            results.append({
                "block_id": obj.properties.get("block_id"),
                "block_type": obj.properties.get("block_type"),
                "keywords": obj.properties.get("keywords_text", "").split(),
            })
    except Exception as e:
        logger.warning("Failed to fetch all block indexes: %s", e)
    return results


# ── Content operations (TravelFixed / TravelDynamic) ───────────────────────────

async def ingest_block(client, collection_name: str, block_id: str,
                      block_type: str, content: str, agent_id: str, tags: list[str] = None) -> None:
    """Ingest a block's content into its isolated tenant (Async)."""
    collection = client.collections.get(collection_name)
    await _ensure_tenant_async(collection, agent_id)
    
    tenant_coll = collection.with_tenant(agent_id)
    await tenant_coll.data.insert({
        "content": content,
        "block_id": block_id,
        "agent_id": agent_id,
        "block_type": block_type,
        "tags": tags or [],
        "updated_at": datetime.now(timezone.utc),
    })
    logger.debug("Block '%s' ingested for tenant '%s' in '%s'.", block_id, agent_id, collection_name)


async def delete_block_vectors(client, block_id: str, agent_id: str) -> None:
    """Delete all vectors for a block from the local tenant (Async)."""
    collection = client.collections.get("TravelDynamic")
    tenant_coll = collection.with_tenant(agent_id)
    await tenant_coll.data.delete_many(
        where=Filter.by_property("block_id").equal(block_id)
    )
    logger.debug("Weaviate vectors deleted for block '%s' (tenant '%s').", block_id, agent_id)


def setup_collections() -> None:
    """Synchronous init (used once at startup)."""
    init_all_schemas()


if __name__ == "__main__":
    setup_collections()
