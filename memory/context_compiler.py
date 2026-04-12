"""
context_compiler.py — Async-first working context builder.

Fixes applied:
- [Fix 3] Filter [DELETED_BLOCK] zombie blocks before injecting into Gemini prompt.
- [Fix 4] Parallelize all Letta HTTP reads natively with asyncio.gather().
          Previously: Sequential blocking calls, then asyncio.to_thread wrappers.
          Now: Native non-blocking async calls via AsyncLetta.
"""

import asyncio
from typing import Optional
from logger import get_logger

logger = get_logger(__name__)


async def get_core_block_content(agent_id: str, label: str) -> Optional[str]:
    """
    Fetch the content of a core memory block from Letta Cloud API (Async).
    Returns None if the block is missing (404), or empty string on other errors.
    """
    from .letta_cloud_client import get_letta_client_async
    letta = get_letta_client_async()
    try:
        block = await letta.agents.blocks.retrieve(label, agent_id=agent_id)
        if block.value is not None:
            return block.value
    except Exception as e:
        if "404" in str(e):
            return None
        logger.warning("Could not read block '%s' from Letta API (Async): %s", label, e)
    return ""


async def _fetch_block_async(agent_id: str, block: dict) -> tuple[dict, str]:
    """
    Fetch a single block's content (native async).
    Returns (block_meta, content) tuple.
    """
    content = await get_core_block_content(agent_id, block["id"])
    return block, content or ""


async def compile_working_context(
    agent_id: str, relevant_blocks: list[dict], query: str = ""
) -> str:
    """
    Assemble the final Working Context string to inject into the Gemini prompt.

    Optimizations vs previous version:
    - All Letta HTTP calls run in parallel (asyncio.gather + to_thread).
    - Zombie blocks (value starts with '[DELETED') are filtered out silently.
    """
    if not relevant_blocks:
        return ""

    # Parallel fetch — all blocks fetched concurrently from Letta Cloud
    fetch_tasks = [_fetch_block_async(agent_id, b) for b in relevant_blocks]
    results: list[tuple[dict, str]] = await asyncio.gather(*fetch_tasks)

    context_parts = []
    injected = 0

    for block, content in results:
        # Fix 3: Skip zombie / soft-deleted blocks
        if content.strip().startswith("[DELETED"):
            logger.debug("Block '%s' skipped (soft-deleted).", block["id"])
            continue

        context_parts.append(f"--- BLOCK: {block['label'].upper()} ({block['type']}) ---")
        context_parts.append(content)
        context_parts.append("")  # blank line separator
        injected += 1

    logger.debug(
        "Working context compiled: %d/%d blocks injected.",
        injected, len(relevant_blocks),
    )
    return "\n".join(context_parts)
