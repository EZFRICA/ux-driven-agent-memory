"""
DLL Manager — Weaviate-backed version.
Implements the Bidirectional Metadata Jump (BMJ) algorithm for memory routing.
Uses Weaviate BlockIndex for vector search instead of local embeddings.
DLL structure (prev/next pointers) stored in metadata_links.json.
"""

import json
import os
from datetime import datetime
from typing import Optional
from config import METADATA_FILE, USER_ID, MAX_DYNAMIC_BLOCKS, FIXED_BLOCKS
from logger import get_logger

logger = get_logger(__name__)

# ── Adaptive certainty thresholds by block type ───────────────────────────────
# Certainty: 0.0 (opposite) to 1.0 (identical) — maps to old similarity scores
CERTAINTY_THRESHOLDS = {
    "fondamental": 0.52,  # Profile/prefs — low threshold, frequently above
    "projet":      0.55,  # Itinerary — neutral threshold
    "temp":        0.60,  # Session — high threshold, jump to TAIL often
}

# Minimum certainty for a block to be included in the working context
MIN_RELEVANCE_CERTAINTY = 0.50


def init_dll() -> dict:
    """
    Initialize the Living DLL V2 with 4 fixed core blocks.
    Vectors are managed by Weaviate BlockIndex — no local embedding computation.
    Ingests initial keyword vectors into Weaviate.
    """
    logger.debug("Initializing DLL V2 — ingesting keywords into Weaviate BlockIndex...")

    dll = {
        "user_id": USER_ID,
        "agent_id": None,
        "head_id": "current_session",
        "tail_id": "traveler_profile",
        "fixed_blocks": FIXED_BLOCKS,
        "dynamic_block_count": 0,
        "dynamic_block_max": MAX_DYNAMIC_BLOCKS,
        "created_at": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        "nodes": {
            "current_session": {
                "id": "current_session",
                "label": "Current Session",
                "letta_block_label": "current_session",
                "weaviate_collection": "TravelFixed",
                "type": "temp",
                "is_fixed": True,
                "created_by": "system",
                "keywords": ["session", "search", "hotel", "options"],
                "active": True,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "prev": None,
                "next": "active_trip"
            },
            "active_trip": {
                "id": "active_trip",
                "label": "Active Trip & Reservations",
                "letta_block_label": "active_trip",
                "weaviate_collection": "TravelFixed",
                "type": "projet",
                "is_fixed": True,
                "created_by": "system",
                "keywords": ["itinerary", "reservations", "flights", "trains", "stages", "planning"],
                "active": True,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "prev": "current_session",
                "next": "traveler_preferences"
            },
            "traveler_preferences": {
                "id": "traveler_preferences",
                "label": "Traveler Preferences",
                "letta_block_label": "traveler_preferences",
                "weaviate_collection": "TravelFixed",
                "type": "fondamental",
                "is_fixed": True,
                "created_by": "system",
                "keywords": ["preferences", "accommodation", "transport", "budget", "rules", "must-have"],
                "active": True,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "prev": "active_trip",
                "next": "traveler_profile"
            },
            "traveler_profile": {
                "id": "traveler_profile",
                "label": "Traveler Profile & Documents",
                "letta_block_label": "traveler_profile",
                "weaviate_collection": "TravelFixed",
                "type": "fondamental",
                "is_fixed": True,
                "created_by": "system",
                "keywords": ["identity", "documents", "passport", "insurance", "nationality"],
                "active": True,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "prev": "traveler_preferences",
                "next": None
            }
        }
    }

    # Ingest keyword vectors into Weaviate BlockIndex
    try:
        from memory.weaviate_cloud_client import get_weaviate_client, setup_collections, upsert_block_index
        client = get_weaviate_client()
        try:
            setup_collections(client)
            for node_id, node in dll["nodes"].items():
                agent_id = dll.get("agent_id") or "default_agent"
                upsert_block_index(client, agent_id, node_id, node["keywords"], node["type"])
            logger.info("All 4 fixed blocks indexed in Weaviate BlockIndex.")
        finally:
            client.close()
    except Exception as e:
        logger.warning("Weaviate indexing failed during init (non-critical): %s", e)

    save_dll(dll)
    return dll


def load_dll() -> dict:
    """Load the DLL state from disk. Initializes a fresh DLL if no file exists."""
    if not os.path.exists(METADATA_FILE):
        return init_dll()
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dll(dll: dict) -> None:
    """Persist the DLL state to disk (JSON)."""
    dll["last_modified"] = datetime.now().isoformat()
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(dll, f, ensure_ascii=False, indent=2)


def get_head_threshold(dll: dict) -> float:
    """Return the adaptive certainty threshold based on the HEAD node type."""
    head_node = dll["nodes"][dll["head_id"]]
    return CERTAINTY_THRESHOLDS.get(head_node["type"], 0.55)


def search_memory(query: str, dll: dict) -> list[dict]:
    """
    Bidirectional Metadata Jump (BMJ) — powered by Weaviate near_text.

    STEP 1 — Query Weaviate BlockIndex for all blocks' relevance scores.
    STEP 2 — Test the HEAD block certainty vs adaptive threshold.
    STEP 3 — Direction decision:
        certainty(HEAD) < THRESHOLD[HEAD.type] → prioritize TAIL blocks (fondamental)
        certainty(HEAD) ≥ THRESHOLD[HEAD.type] → keep original Weaviate ranking

    Returns: top-3 active blocks sorted by certainty (descending).
    """
    logger.debug("DLL Search | query='%s'", query)

    # STEP 1: Query Weaviate for all block scores
    try:
        from memory.weaviate_cloud_client import get_weaviate_client, search_block_index
        client = get_weaviate_client()
        try:
            agent_id = dll.get("agent_id") or "default_agent"
            weaviate_results = search_block_index(client, agent_id, query, limit=12)
        finally:
            client.close()
    except Exception as e:
        logger.error("Weaviate search failed: %s. Returning empty.", e)
        return []

    # Build certainty map: block_id → certainty
    certainty_map = {r["block_id"]: r["certainty"] for r in weaviate_results}

    # STEP 2: Test HEAD certainty
    head_id = dll["head_id"]
    head_certainty = certainty_map.get(head_id, 0.0)
    threshold = get_head_threshold(dll)

    logger.debug("HEAD='%s' | certainty=%.3f | threshold[%s]=%.2f",
                 head_id, head_certainty, dll["nodes"][head_id]["type"], threshold)

    # STEP 3: Direction decision
    if head_certainty < threshold:
        logger.debug("%.3f < %.2f — JUMP to TAIL (traverse via .prev)", head_certainty, threshold)
        traversal_order = _tail_to_head_order(dll)
    else:
        logger.debug("%.3f >= %.2f — Traversal HEAD → TAIL (via .next)", head_certainty, threshold)
        traversal_order = _head_to_tail_order(dll)

    # STEP 4: Injection & Override Logic
    # 1. Fixed blocks (4 for Travel) are ALWAYS injected.
    # 2. Force-Inject requested dynamic blocks ("active" = True).
    # 3. BMJ Fill: Use non-checked dynamic blocks if they meet the semantic threshold.
    # The absolute maximum blocks we can ever send is 12.
    
    forced_blocks = []
    dynamic_candidates = []
    
    for node_id in traversal_order:
        node = dll["nodes"][node_id]
        certainty = certainty_map.get(node_id, 0.0)
        
        is_fixed = node.get("is_fixed", False)
        is_active = node.get("active", True)
        
        # Fixed blocks or user-forced dynamic blocks
        if is_fixed or is_active:
            forced_blocks.append((certainty, node))
        else:
            # BMJ candidates: Unforced dynamic blocks
            if certainty >= MIN_RELEVANCE_CERTAINTY:
                dynamic_candidates.append((certainty, node))
                
    # Sort candidates by certitude
    dynamic_candidates.sort(key=lambda x: x[0], reverse=True)
    
    top_blocks = [item[1] for item in forced_blocks]
    
    for certainty, node in dynamic_candidates:
        # Fill context up to the hard maximum (12 total blocks)
        if len(top_blocks) < 12 and certainty >= threshold:
            top_blocks.append(node)

    for idx, b in enumerate(top_blocks):
        logger.debug("[%d] Selected: '%s' (type=%s)", idx + 1, b["label"], b["type"])

    return top_blocks


def move_to_front(block_id: str, dll: dict) -> dict:
    """Move the selected node to HEAD position (O(1) access on next query)."""
    if dll["head_id"] == block_id:
        return dll

    nodes = dll["nodes"]
    target = nodes[block_id]
    prev_id, next_id = target["prev"], target["next"]

    if prev_id:
        nodes[prev_id]["next"] = next_id
    if next_id:
        nodes[next_id]["prev"] = prev_id
    if dll["tail_id"] == block_id:
        dll["tail_id"] = prev_id

    old_head = dll["head_id"]
    nodes[old_head]["prev"] = block_id
    target["prev"] = None
    target["next"] = old_head
    dll["head_id"] = block_id

    target["access_count"] += 1
    target["last_accessed"] = datetime.now().isoformat()
    logger.debug("MTF: '%s' → HEAD. Order: %s", block_id, _head_to_tail_order(dll))
    return dll


def toggle_block(block_id: str, state: bool, dll: dict) -> dict:
    """Enable or disable a block (False = Let BMJ decide, True = Force Inject)."""
    if block_id in dll["nodes"]:
        dll["nodes"][block_id]["active"] = state
        status = "enabled" if state else "disabled"
        logger.debug("TOGGLE: '%s' %s", block_id, status)
    else:
        logger.warning("Block '%s' not found in DLL.", block_id)
    return dll


def update_node_keywords(block_id: str, keywords: list[str], dll: dict) -> dict:
    """Update a node's keywords and re-index in Weaviate BlockIndex."""
    if block_id not in dll["nodes"]:
        return dll
    dll["nodes"][block_id]["keywords"] = keywords

    # Re-index in Weaviate
    try:
        from memory.weaviate_cloud_client import get_weaviate_client, upsert_block_index
        client = get_weaviate_client()
        try:
            agent_id = dll.get("agent_id") or "default_agent"
            block_type = dll["nodes"][block_id]["type"]
            upsert_block_index(client, agent_id, block_id, keywords, block_type)
            logger.debug("Node '%s' re-indexed in Weaviate with new keywords.", block_id)
        finally:
            client.close()
    except Exception as e:
        logger.warning("Weaviate re-index failed for '%s': %s", block_id, e)

    return dll


def get_all_nodes(dll: dict) -> list[dict]:
    """Return all nodes in HEAD → TAIL order for UI display."""
    nodes = []
    current = dll["head_id"]
    visited = set()
    while current and current not in visited:
        visited.add(current)
        nodes.append(dll["nodes"][current])
        current = dll["nodes"][current]["next"]
    return nodes


def _head_to_tail_order(dll: dict) -> list[str]:
    """Traverse the DLL from HEAD to TAIL via .next links."""
    order, current = [], dll["head_id"]
    visited = set()
    while current and current not in visited:
        visited.add(current)
        order.append(current)
        current = dll["nodes"][current]["next"]
    return order


def _tail_to_head_order(dll: dict) -> list[str]:
    """Traverse the DLL from TAIL to HEAD via .prev links."""
    order, current = [], dll["tail_id"]
    visited = set()
    while current and current not in visited:
        visited.add(current)
        order.append(current)
        current = dll["nodes"][current]["prev"]
    return order


if __name__ == "__main__":
    import sys
    if "--init" in sys.argv:
        init_dll()
        logger.info("metadata_links.json initialized successfully.")
