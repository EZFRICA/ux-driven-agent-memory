"""
Weaviate Cloud Client — operations layer.
Schema definitions are in schema.py.
This module handles CRUD operations on BlockIndex, TravelFixed, and TravelDynamic.
"""

from datetime import datetime, timezone
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.util import generate_uuid5
from memory.schema import get_weaviate_client, init_all_schemas
from logger import get_logger

logger = get_logger(__name__)


# ── BlockIndex operations (DLL routing) ────────────────────────────────────────

def upsert_block_index(client, block_id: str, keywords: list[str], block_type: str, agent_id: str) -> None:
    """
    Insert or update a block's keyword vector in the BlockIndex collection.
    The 'keywords_text' field is auto-vectorized by Weaviate for near_text search.
    Uses deterministic UUID based on agent_id and block_id for upsert behavior.
    """
    collection = client.collections.get("BlockIndex")
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
        # Try update first (existing block)
        collection.data.update(uuid=obj_uuid, properties=properties)
        logger.debug("BlockIndex updated: '%s' for agent '%s'.", block_id, agent_id)
    except Exception:
        # Not found → insert
        collection.data.insert(uuid=obj_uuid, properties=properties)
        logger.debug("BlockIndex inserted: '%s' for agent '%s'.", block_id, agent_id)


def search_block_index(client, query: str, agent_id: str, limit: int = 12) -> list[dict]:
    """
    Semantic search on BlockIndex using near_text, scoped to the agent_id.
    Returns: [{block_id, block_type, certainty}, ...] sorted by relevance (highest first).
    """
    collection = client.collections.get("BlockIndex")
    response = collection.query.near_text(
        query=query,
        limit=limit,
        target_vector="keywords_text",
        filters=Filter.by_property("agent_id").equal(agent_id),
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


def delete_block_index(client, block_id: str, agent_id: str) -> None:
    """Remove a block from the BlockIndex collection."""
    collection = client.collections.get("BlockIndex")
    obj_uuid = generate_uuid5(f"{agent_id}_{block_id}")
    try:
        collection.data.delete_by_id(obj_uuid)
        logger.debug("BlockIndex deleted: '%s'.", block_id)
    except Exception as e:
        logger.warning("BlockIndex delete failed for '%s': %s", block_id, e)

def fetch_all_block_indexes(client, agent_id: str) -> list[dict]:
    """
    Fetch all blocks from the BlockIndex collection for a specific agent.
    """
    collection = client.collections.get("BlockIndex")
    results = []
    try:
        response = collection.query.fetch_objects(
            filters=Filter.by_property("agent_id").equal(agent_id),
            limit=100
        )
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

def ingest_block(client, collection_name: str, block_id: str,
                 block_type: str, content: str, agent_id: str, tags: list[str] = None) -> None:
    """Ingest a block's content into its Weaviate collection (TravelFixed or TravelDynamic)."""
    collection = client.collections.get(collection_name)
    collection.data.insert({
        "content": content,
        "block_id": block_id,
        "agent_id": agent_id,
        "block_type": block_type,
        "tags": tags or [],
        "updated_at": datetime.now(timezone.utc),
    })
    logger.debug("Block '%s' ingested into '%s' for agent '%s'.", block_id, collection_name, agent_id)


def delete_block_vectors(client, block_id: str, agent_id: str) -> None:
    """
    Delete all vectors for a block from TravelDynamic.
    Fixed blocks (TravelFixed) are never deleted.
    """
    collection = client.collections.get("TravelDynamic")
    collection.data.delete_many(
        where=Filter.by_property("block_id").equal(block_id) & Filter.by_property("agent_id").equal(agent_id)
    )
    logger.debug("Weaviate vectors deleted for block '%s' (agent '%s').", block_id, agent_id)


def setup_collections(client=None) -> None:
    """Alias for init_all_schemas() — backward compatibility."""
    init_all_schemas()


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        logger.info("Setting up Weaviate collections...")
        init_all_schemas()
