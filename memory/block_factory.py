from datetime import datetime
from memory.dll_manager import update_node_keywords, save_dll
from memory.weaviate_cloud_client import (
    get_weaviate_client_async,
    upsert_block_index,
    delete_block_index,
    ingest_block,
    delete_block_vectors,
)
from logger import get_logger

logger = get_logger(__name__)


def insert_node_by_type(block_type: str, new_node: dict, dll: dict) -> dict:
    """
    Insert a new node at the position matching its semantic type:
        temp        → HEAD (most recent context)
        projet      → after HEAD (mid-list, active planning)
        fondamental → before TAIL (permanent knowledge, always reachable)
    """
    nodes = dll["nodes"]

    if block_type == "temp":
        old_head = dll["head_id"]
        new_node["next"] = old_head
        new_node["prev"] = None
        if old_head:
            nodes[old_head]["prev"] = new_node["id"]
        dll["head_id"] = new_node["id"]

    elif block_type == "fondamental":
        old_tail = dll["tail_id"]
        if old_tail:
            prev_to_tail = nodes[old_tail]["prev"]
            new_node["next"] = old_tail
            new_node["prev"] = prev_to_tail
            nodes[old_tail]["prev"] = new_node["id"]
            if prev_to_tail:
                nodes[prev_to_tail]["next"] = new_node["id"]
            else:
                dll["head_id"] = new_node["id"]
        else:
            dll["head_id"] = dll["tail_id"] = new_node["id"]

    else:  # projet — insert after HEAD
        head = dll["head_id"]
        if head:
            next_to_head = nodes[head]["next"]
            new_node["prev"] = head
            new_node["next"] = next_to_head
            nodes[head]["next"] = new_node["id"]
            if next_to_head:
                nodes[next_to_head]["prev"] = new_node["id"]
            else:
                dll["tail_id"] = new_node["id"]
        else:
            dll["head_id"] = dll["tail_id"] = new_node["id"]

    nodes[new_node["id"]] = new_node
    return dll


async def delete_block_stitching(block_id: str, dll: dict) -> dict:
    """
    Remove a dynamic block from the DLL and re-stitch its neighbors.
    Also removes it from Weaviate BlockIndex and TravelDynamic.
    """
    nodes = dll["nodes"]

    if block_id not in nodes:
        raise ValueError(f"Block '{block_id}' does not exist.")

    target = nodes[block_id]

    if target.get("is_fixed", False):
        raise ValueError(f"Block '{block_id}' is a fixed block and cannot be deleted.")

    prev_id, next_id = target["prev"], target["next"]

    if prev_id:
        nodes[prev_id]["next"] = next_id
    if next_id:
        nodes[next_id]["prev"] = prev_id

    if dll["head_id"] == block_id:
        dll["head_id"] = next_id
    if dll["tail_id"] == block_id:
        dll["tail_id"] = prev_id

    del nodes[block_id]
    dll["dynamic_block_count"] = max(0, dll["dynamic_block_count"] - 1)

    # Clean up Weaviate
    try:
        async with get_weaviate_client_async() as client:
            agent_id = dll.get("agent_id")
            if not agent_id:
                raise ValueError("Agent ID is not defined in the DLL.")
            await delete_block_index(client, block_id, agent_id)
            await delete_block_vectors(client, block_id, agent_id)
    except Exception as e:
        logger.warning("Weaviate cleanup failed for '%s': %s", block_id, e)

    logger.debug("Block '%s' removed from DLL. Dynamic count: %d", block_id, dll["dynamic_block_count"])
    return dll


async def create_dynamic_block(
    block_id: str,
    label: str,
    block_type: str,
    initial_content: str,
    keywords: list[str],
    created_by: str,
    dll: dict,
    letta_client,
    wcd_client,
) -> dict:
    """
    Create a new dynamic block using ACID-like principles.
    Enforces strict synchronization: Letta -> Weaviate -> Local JSON.
    If Letta or Weaviate fails, the operation aborts and local state remains untouched.
    """
    if dll["dynamic_block_count"] >= dll["dynamic_block_max"]:
        raise ValueError(
            f"Dynamic block limit reached ({dll['dynamic_block_max']} blocks maximum)."
        )

    if block_id in dll["nodes"]:
        raise ValueError(f"Block '{block_id}' already exists.")

    agent_id = dll.get("agent_id")
    if not agent_id:
        raise ValueError("Agent ID is not defined in the DLL.")

    # 1. Sync to Letta Cloud (Source of Truth for creation/content)
    if letta_client:
        try:
            await letta_client.append_block(agent_id, block_id, initial_content, block_type)
            logger.debug("Letta sync: block '%s' appended.", block_id)
        except Exception as e:
            logger.error("Letta sync failed for '%s'. Aborting creation. Error: %s", block_id, e)
            raise RuntimeError(f"Failed to create block in Letta Cloud. Aborting. Error: {e}")

    # 2. Sync to Weaviate Cloud (Search Index & Content Backup)
    if wcd_client:
        try:
            async with wcd_client.get_weaviate_client_async() as client:
                # 2A: Index Keywords
                await upsert_block_index(client, block_id, keywords, block_type, agent_id)
                # 2B: Ingest initial Content
                await wcd_client.ingest_block(client, "TravelDynamic", block_id, block_type, initial_content, agent_id)
                logger.debug("Weaviate content & index sync: block '%s' ingested.", block_id)
        except Exception as e:
            # Note: Weaviate failed. Strictly speaking, we should delete from Letta here to rollback completely.
            logger.error("Weaviate sync failed for '%s'. Assuming Letta succeeded but Weaviate failed. Error: %s", block_id, e)
            try:
                if letta_client:
                    await letta_client.delete_block(agent_id, block_id)
                    logger.warning("Rollback: Letta block '%s' deleted after Weaviate failure.", block_id)
            except Exception as rollback_err:
                logger.error(
                    "Rollback FAILED for block '%s': %s. "
                    "Letta and Weaviate may be out of sync — manual cleanup required.",
                    block_id, rollback_err,
                )
            raise RuntimeError(f"Failed to create block in Weaviate. Rollback triggered. Error: {e}")

    # 3. Update Local DLL State (Only if external DBs succeed)
    new_node = {
        "id": block_id,
        "label": label,
        "letta_block_label": block_id,
        "weaviate_collection": "TravelDynamic",
        "type": block_type,
        "is_fixed": False,
        "created_by": created_by,
        "keywords": keywords,
        "active": True,
        "access_count": 0,
        "last_accessed": None,
        "last_modified": datetime.now().isoformat(),
        "prev": None,
        "next": None,
    }

    dll = insert_node_by_type(block_type, new_node, dll)
    dll["dynamic_block_count"] += 1
    save_dll(dll)
    logger.debug("Dynamic block '%s' fully created and persisted to JSON (type=%s).", block_id, block_type)
    
    return dll


async def update_block_content(
    block_id: str,
    new_content: str,
    new_keywords: list[str],
    dll: dict,
    letta_client,
    wcd_client,
) -> dict:
    """
    Cascade Update (ACID-like) — synchronizes across all 3 stores:
        1. Letta Cloud Core Memory (Source of Truth)
        2. Weaviate BlockIndex (re-index keywords) & Weaviate content
        3. Local JSON (Last, only if 1 and 2 succeed)
    """
    nodes = dll["nodes"]
    if block_id not in nodes:
        raise ValueError(f"Block '{block_id}' not found in DLL.")

    node = nodes[block_id]
    agent_id = dll.get("agent_id")
    if not agent_id:
        raise ValueError("Agent ID is not defined in the DLL.")

    # 1. Update Letta Core Memory
    if letta_client:
        try:
            await letta_client.update_block(agent_id, block_id, new_content)
            logger.debug("Cascade: Letta block '%s' updated.", block_id)
        except Exception as e:
            logger.error("Cascade: Letta update failed for '%s'. Aborting sync. Error: %s", block_id, e)
            raise RuntimeError(f"Failed to update block in Letta Cloud. Aborting. Error: {e}")

    # 2. Update Weaviate (Content and Keywords)
    if wcd_client:
        try:
            async with wcd_client.get_weaviate_client_async() as client:
                # 2A: Re-ingest content (delete old, insert new)
                collection = "TravelFixed" if node.get("is_fixed") else "TravelDynamic"
                
                if collection == "TravelDynamic":
                    await wcd_client.delete_block_vectors(client, block_id, agent_id)
                await wcd_client.ingest_block(client, collection, block_id, node["type"], new_content, agent_id)
                logger.debug("Cascade: Weaviate block '%s' re-ingested into '%s'.", block_id, collection)
        except Exception as e:
            logger.error("Cascade: Weaviate sync failed for '%s'. Error: %s", block_id, e)
            raise RuntimeError(f"Failed to update block in Weaviate. Letta may be out of sync. Error: {e}")

    # 3. Update Local DLL State (Only if external APIs succeeded)
    # The JSON node modification happens at the very end
    dll = await update_node_keywords(block_id, new_keywords, dll)
    # Ensure node exists in our reference after update
    node = dll["nodes"][block_id]
    node["last_modified"] = datetime.now().isoformat()
    save_dll(dll)
    logger.debug("Cascade: block '%s' metadata saved to JSON.", block_id)
    
    return dll
