import os
import sys

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.dll_manager import load_dll, get_all_nodes
from memory.weaviate_cloud_client import get_weaviate_client, upsert_block_index, ingest_block
from memory.context_compiler import get_core_block_content
from logger import get_logger

logger = get_logger(__name__)

def sync_all():
    """Sync all blocks from metadata_links.json and Letta into Weaviate."""
    dll = load_dll()
    agent_id = dll.get("agent_id")
    
    if not agent_id:
        logger.error("No agent_id found in DLL metadata. Run --create-agent first.")
        return

    client = get_weaviate_client()
    try:
        nodes = get_all_nodes(dll)
        logger.info(f"Starting sync for {len(nodes)} blocks...")
        
        for node in nodes:
            b_id = node["id"]
            kw = node.get("keywords", [])
            b_type = node.get("type", "projet")
            
            # 1. Sync keywords to BlockIndex (for routing)
            upsert_block_index(client, b_id, kw, b_type, agent_id)
            
            # 2. Sync content to TravelFixed/TravelDynamic (for archival)
            content = get_core_block_content(agent_id, b_id)
            coll = "TravelFixed" if node.get("is_fixed") else "TravelDynamic"
            
            if content.strip():
                ingest_block(client, coll, b_id, b_type, content, agent_id)
                logger.info(f"Synced '{b_id}' to Weaviate.")
            else:
                logger.debug(f"Skipping content for '{b_id}' (empty).")

        logger.info("Sync complete! DLL search (BMJ) will now work with Weaviate.")
        
    finally:
        client.close()

if __name__ == "__main__":
    sync_all()
