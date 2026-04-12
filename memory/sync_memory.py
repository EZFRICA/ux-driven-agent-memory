import os
import sys
import asyncio

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.dll_manager import load_dll, get_all_nodes
from memory.weaviate_cloud_client import get_weaviate_client_async, upsert_block_index, ingest_block
from memory.context_compiler import get_core_block_content
from logger import get_logger

logger = get_logger(__name__)

async def sync_all():
    """Sync all blocks from metadata_links.json and Letta into Weaviate (Async)."""
    dll = await load_dll()
    agent_id = dll.get("agent_id")
    
    if not agent_id:
        logger.error("No agent_id found in DLL metadata. Run --create-agent first.")
        return

    async with get_weaviate_client_async() as client:
        nodes = get_all_nodes(dll)
        logger.info(f"Starting async sync for {len(nodes)} blocks...")
        
        for node in nodes:
            b_id = node["id"]
            kw = node.get("keywords", [])
            b_type = node.get("type", "projet")
            
            # 1. Sync keywords to BlockIndex (for routing) - with tenant isolation
            await upsert_block_index(client, b_id, kw, b_type, agent_id)
            
            # 2. Sync content to TravelFixed/TravelDynamic (for archival)
            # This is still a sync HTTP call in context_compiler, but scoped in a thread pool later
            content = await get_core_block_content(agent_id, b_id)
            coll = "TravelFixed" if node.get("is_fixed") else "TravelDynamic"
            
            if content and content.strip():
                await ingest_block(client, coll, b_id, b_type, content, agent_id)
                logger.info(f"Synced '{b_id}' to Weaviate isolated tenant.")
            else:
                logger.debug(f"Skipping content for '{b_id}' (empty).")

        logger.info("Async Sync complete! DLL search (BMJ) will now work with multi-tenancy.")

if __name__ == "__main__":
    asyncio.run(sync_all())
