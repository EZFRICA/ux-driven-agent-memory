import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from letta_client import AsyncLetta
from config import LETTA_API_KEY, LETTA_BASE_URL
from logger import get_logger

logger = get_logger(__name__)
_letta_client_async = None

def get_letta_client_async() -> AsyncLetta:
    """Lazy-initialized Async Letta client singleton."""
    global _letta_client_async
    if _letta_client_async is None:
        if not LETTA_API_KEY:
             raise ValueError("LETTA_API_KEY is missing from the environment. Check your .env file.")
        _letta_client_async = AsyncLetta(
            api_key=LETTA_API_KEY,
            base_url=LETTA_BASE_URL,
        )
    return _letta_client_async

INITIAL_FIXED_BLOCKS = {
    "traveler_profile": {
        "label": "My Profile",
        "type": "fondamental",
        "initial_value": "",
    },
    "traveler_preferences": {
        "label": "My Preferences",
        "type": "fondamental",
        "initial_value": "",
    },
    "active_trip": {
        "label": "Active Trip",
        "type": "projet",
        "initial_value": "",
    },
    "current_session": {
        "label": "Current Session",
        "type": "temp",
        "initial_value": "",
    },
}


async def create_travel_agent() -> str:
    """
    Create a Letta Cloud agent with the 4 fixed core memory blocks (Async).
    """
    letta = get_letta_client_async()
    agent_state = await letta.agents.create(
        model="my-gemini-key/gemini-flash-lite-latest",
        embedding="my-gemini-key/gemini-embedding-2-preview",
        memory_blocks=[
            {
                "label": block_id,
                "value": cfg["initial_value"],
                "description": f"{cfg['label']} — type={cfg['type']}.",
            }
            for block_id, cfg in INITIAL_FIXED_BLOCKS.items()
        ],
        tools=[
            "archival_memory_search",
            "archival_memory_insert",
            "core_memory_replace",
        ],
        system=(
            "You are a travel planning agent. "
            "CRITICAL RULE: before each response, the DLL Manager has already selected "
            "the relevant memory blocks. You must use ONLY the blocks listed in the Working Context. "
            "End every response with: [Memory: <blocks used> | DLL: <X>/12 blocks]"
        ),
    )
    logger.info("Letta agent created with ID: %s", agent_state.id)
    return agent_state.id


async def update_block(agent_id: str, block_label: str, new_content: str) -> None:
    """Update a core memory block value via the Letta SDK (Async)."""
    letta = get_letta_client_async()
    await letta.agents.blocks.update(
        block_label,
        agent_id=agent_id,
        value=new_content,
    )


async def append_block(agent_id: str, block_label: str, content: str, block_type: str = "projet") -> None:
    """
    Create a new dynamic block in Letta Cloud and attach it (Async).
    """
    logger.debug("Creating block '%s' and attaching to agent %s...", block_label, agent_id)
    letta = get_letta_client_async()
    try:
        # 1. Create the block
        new_block = await letta.blocks.create(label=block_label, value=content, limit=2000)
        # 2. Attach
        await letta.agents.blocks.attach(block_id=new_block.id, agent_id=agent_id)
        logger.debug("Letta block '%s' created and attached successfully.", block_label)
    except Exception as e:
        if "409" in str(e) or "UniqueViolationError" in str(e):
            logger.debug("Block '%s' already exists for this agent. Falling back to update.", block_label)
            await update_block(agent_id, block_label, content)
        else:
            logger.error("Letta: failed to create/attach block '%s': %s", block_label, e)
            raise e


async def delete_block(agent_id: str, block_label: str) -> None:
    """Logical block deletion (Async)."""
    letta = get_letta_client_async()
    try:
        await letta.agents.blocks.update(
            block_label,
            agent_id=agent_id,
            value="[DELETED_BLOCK]",
        )
        logger.debug("Letta block '%s' cleared (logical delete).", block_label)
    except Exception as e:
        logger.error("Letta: failed to delete block '%s': %s", block_label, e)


async def search_archival(agent_id: str, query: str, block_label: str, limit: int = 5) -> list[str]:
    """Semantic search in Letta archival memory (Async)."""
    letta = get_letta_client_async()
    results = await letta.agents.archival_memory.list(
        agent_id=agent_id,
        query=query,
        limit=limit,
    )
    return [r.text for r in results if block_label in (r.metadata or {}).get("label", "")]


async def send_message(agent_id: str, working_context: str, user_query: str) -> str:
    """Send a message to the Letta agent (Async)."""
    letta = get_letta_client_async()
    full_message = f"[WORKING CONTEXT]\n{working_context}\n\n[QUERY]\n{user_query}"
    response = await letta.agents.messages.create(
        agent_id=agent_id,
        messages=[{"role": "user", "content": full_message}],
    )
    for msg in response.messages:
        if msg.message_type == "assistant_message":
            return msg.content
    return ""


if __name__ == "__main__":
    import sys
    import asyncio
    from dll_manager import load_dll, save_dll
    if "--create-agent" in sys.argv:
        logger.info("Creating travel planning agent...")
        agent_id = asyncio.run(create_travel_agent())
        dll = asyncio.run(load_dll())
        dll["agent_id"] = agent_id
        save_dll(dll)
        logger.info("Agent ID %s saved to DLL metadata.", agent_id)
