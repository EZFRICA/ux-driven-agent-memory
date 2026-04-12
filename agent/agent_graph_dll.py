import os
import sys
import json

# Add project root to path for memory module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY, USER_ID
from memory.dll_manager import search_memory, load_dll, save_dll, move_to_front, get_all_nodes
from memory.context_compiler import compile_working_context, get_core_block_content
from memory.block_detector import detect_new_block_opportunity
from memory.block_factory import create_dynamic_block, update_block_content, delete_block_stitching
from memory.letta_cloud_client import append_block, update_block
from memory import weaviate_cloud_client as wcd_client
from memory import letta_cloud_client as letta_client
from logger import get_logger

logger = get_logger(__name__)

GEMINI_MODEL = "gemini-flash-lite-latest"

# Main LLM — conversational, slightly creative
_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.7,
    google_api_key=GEMINI_API_KEY,
)

# Extractor LLM — deterministic, used for memory write-back only
_extractor_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.0,
    google_api_key=GEMINI_API_KEY,
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    agent_id: str
    search_enabled: bool  # New toggle for Google Search
    memory_only_mode: bool  # Toggle for Memory-only mode (ignore history)
    strict_manual_mode: bool
    needs_new_block: str
    proposed_block_config: dict


async def _extract_and_update_memory(
    agent_id: str,
    user_query: str,
    agent_response: str,
    relevant_blocks: list[dict],
    dll: dict,
) -> None:
    """
    Memory write-back step — runs after each agent response.

    Uses a deterministic LLM call (temperature=0) to extract factual information
    explicitly shared by the user and writes it back to the relevant Letta blocks.
    New information is appended to existing content, not overwritten.
    Non-critical: failures are logged as warnings and do not interrupt the response.
    """
    if not agent_id:
        return

    # Read current state of each relevant block for deduplication
    block_summaries = []
    for block in relevant_blocks:
        current = await get_core_block_content(agent_id, block["id"])
        block_summaries.append(
            f'- {block["id"]} ({block["type"]}): "{current.strip() or "[empty]"}"'
        )

    # Dynamic instructions based on current DLL blocks
    block_definitions = []
    for b_id, node in dll["nodes"].items():
        block_definitions.append(f"- {b_id}: {node['label']} (Keywords: {', '.join(node.get('keywords', []))})")

    extraction_prompt = f"""You are a memory extraction system for a travel planning assistant.

The user just said:
"{user_query}"

The agent responded:
"{agent_response[:600]}"

Current memory blocks state (ID: content snippet):
{chr(10).join(block_summaries)}

AVAILABLE BLOCKS TO UPDATE:
{chr(10).join(block_definitions)}

TASK: Extract ONLY new factual information the user explicitly stated or confirmed.
Rules:
- Do NOT infer, guess, or include agent suggestions.
- Do NOT repeat information already present in the blocks.
- New info should be appended (short sentences or bullet points).
- If nothing new was shared, return an empty string for that block.

Return ONLY a valid JSON object, keys must match the block IDs above:
{{
  "block_id_1": "new info here",
  "block_id_2": ""
}}
"""

    try:
        extraction_response = await _extractor_llm.ainvoke(
            [HumanMessage(content=extraction_prompt)]
        )
        raw = extraction_response.content
        if isinstance(raw, list):
            raw = "\n".join(
                b.get("text", "") for b in raw
                if isinstance(b, dict) and b.get("type") == "text"
            )

        # Strip markdown code fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        
        try:
            updates: dict = json.loads(raw)
        except json.JSONDecodeError as decode_err:
            logger.warning("Memory extraction parsing failed: %s | Raw response snippet: %s", decode_err, raw[:100])
            return

        for block_id, new_info in updates.items():
            if not new_info or not new_info.strip():
                continue
            
            current = (await get_core_block_content(agent_id, block_id)).strip()
            
            # Smart Clear: remove placeholders like "[No active trip]" if we have new real data
            if current.startswith("[") and current.endswith("]"):
                current = ""
            
            merged = (
                (current + "\n" + new_info.strip()).strip()
                if current
                else new_info.strip()
            )
            
            # Transactional Update (Async Sync Letta + Weaviate)
            node = dll["nodes"].get(block_id)
            if node:
                await update_block_content(
                    block_id, 
                    merged, 
                    node.get("keywords", []), 
                    dll, 
                    letta_client, 
                    wcd_client
                )
                logger.info("Memory write-back (ACID): '%s' updated.", block_id)

    except Exception as e:
        logger.warning("Memory extraction failed (non-critical): %s", e)


async def planner_node_dll(state: AgentState):
    """
    Planner node — core of the LangGraph loop.

    Pipeline:
        1. Extract the latest user query.
        2. DLL vector routing (BMJ) → select relevant memory blocks.
        3. Compile Working Context from Letta Core Memory.
        4. Call Gemini to generate travel response.
        5. Memory write-back: extract and persist new user info to Letta.
        6. Block detector: check for structuring topic opportunities.
        7. Move-To-Front on top-ranked block.
    """
    messages = state["messages"]
    agent_id = state.get("agent_id")

    # 1. Extract last user query
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = str(msg.content)
            break

    # 2. DLL Vector Routing
    dll = await load_dll()
    relevant_blocks = await search_memory(user_query, dll, strict_manual=state.get("strict_manual_mode", False))

    # 3. Compile Working Context from Letta
    context_str = await compile_working_context(agent_id, relevant_blocks, user_query)

    # 4. Build system prompt and call Gemini
    all_nodes = get_all_nodes(dll)
    total_existing = len(all_nodes)
    injected_count = len(relevant_blocks)
    injected_labels = " + ".join(b["id"] for b in relevant_blocks) or "none"

    system_prompt = f"""You are 'Travel Architect', an expert travel planning assistant.

WORKING CONTEXT (DLL filtered — {injected_count}/{total_existing} blocks injected):
{context_str}

INSTRUCTIONS:
- Answer travel requests based on the memory context above.
- If 'Google Search' is enabled, use it to find real-time prices, availability or local events.
- Format responses clearly using markdown.
- Do NOT add a memory footer — the system appends it automatically.
"""
    
    # Configure LLM based on search toggle
    current_llm = _llm
    if state.get("search_enabled"):
        current_llm = _llm.bind(tools=[{"google_search": {}}])

    # Determine message payload based on memory_only_mode
    if state.get("memory_only_mode"):
        # Find the last human message
        last_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        payload = [SystemMessage(content=system_prompt)]
        if last_human_message:
            payload.append(last_human_message)
    else:
        payload = [SystemMessage(content=system_prompt)] + messages

    response = await current_llm.ainvoke(payload)

    # Extract text — Gemini may return string or list of content blocks
    raw = response.content
    if isinstance(raw, str):
        text_content = raw
    elif isinstance(raw, list):
        text_content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    else:
        text_content = str(raw)

    # System-generated footer — always reflects actual DLL state
    footer = f"\n\n---\n`[Memory: {injected_labels} | DLL: {injected_count}/{total_existing} blocks]`"
    text_content = text_content.rstrip() + footer
    logger.debug("Planner response generated (%d chars).", len(text_content))

    # 5. Memory write-back (Extract and update relevant memory blocks)
    await _extract_and_update_memory(agent_id, user_query, text_content, relevant_blocks, dll)

    # 6. Block detector
    raw_history = [{"role": msg.type, "content": str(msg.content)} for msg in messages]
    proposal = detect_new_block_opportunity(raw_history, dll)

    needs_new_block = "False"
    proposed_block_config = {}

    if proposal:
        needs_new_block = "True"
        proposed_block_config = proposal
        logger.info("Block proposal: '%s' (%s).", proposal["proposed_id"], proposal["type"])

    # 7. Move-To-Front
    if relevant_blocks:
        dll = move_to_front(relevant_blocks[0]["id"], dll)
        save_dll(dll)

    msg = AIMessage(content=text_content)
    msg.name = "Planner"

    return {
        "messages": [msg],
        "needs_new_block": needs_new_block,
        "proposed_block_config": proposed_block_config,
    }


def route_supervisor(state: AgentState) -> Literal["__end__"]:
    """Single-node graph — always terminates after the Planner."""
    return END


def create_dll_agent_graph():
    """Build and compile the LangGraph agent workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("Planner", planner_node_dll)
    workflow.add_edge(START, "Planner")
    workflow.add_conditional_edges("Planner", route_supervisor)
    return workflow.compile()
