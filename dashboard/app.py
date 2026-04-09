import streamlit as st
import sys
import os
import asyncio
import re
from datetime import datetime

# Root path alignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.agent_graph_dll import create_dll_agent_graph
from memory.dll_manager import load_dll, save_dll, get_all_nodes, toggle_block, update_node_keywords
from memory.block_factory import create_dynamic_block, update_block_content, delete_block_stitching
from memory.letta_cloud_client import update_block, delete_block
from memory import letta_cloud_client as letta_client
from memory import weaviate_cloud_client as wcd_client
from memory.context_compiler import get_core_block_content

st.set_page_config(page_title="Travel Agent — DLL Dashboard", page_icon="✈️", layout="wide")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []
if "agent_app" not in st.session_state:
    st.session_state.agent_app = create_dll_agent_graph()
if "pending_proposal" not in st.session_state:
    st.session_state.pending_proposal = None
if "memory_facts" not in st.session_state:
    st.session_state.memory_facts = {}
if "last_injected_ids" not in st.session_state:
    st.session_state.last_injected_ids = []
if "sync_version" not in st.session_state:
    st.session_state.sync_version = 0

# Load local DLL state
dll_state = load_dll()
agent_id = dll_state.get("agent_id")

if not agent_id:
    st.error("No Letta agent_id defined. Please run the agent creation script.")
    st.stop()

# --- SIDEBAR : NAVIGATION & OPTIONS ---
with st.sidebar:
    st.title("🧠 UX-Memory")
    st.info(f"Active DLL : {len(get_all_nodes(dll_state))}/12 blocks")
    st.divider()
    
    search_on = st.toggle("🔍 Search Online (Gemini Grounding)", value=False, help="Enables real-time Google search")
    memory_only_on = st.toggle("🧠 Memory-Only Mode", value=False, help="Forces the LLM to rely ONLY on DLL blocks, ignoring chat history")
    strict_manual_on = st.toggle("🚦 Disable BMJ Auto-Retrieval", value=False, help="Demo Mode: Unchecked blocks will NEVER be retrieved automatically.")
    st.divider()
    
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.session_state.pending_proposal = None
        st.rerun()

    if st.button("☁️ Recover Letta Blocks", use_container_width=True, help="Retrieve dynamic blocks stored in Letta that are missing locally."):
        with st.spinner("Fetching missing blocks from Letta..."):
            try:
                # We fetch the agent state to find all blocks attached
                # Using Letta SDK to list blocks (could be via agent state or blocks list)
                try:
                    blocks = letta_client.letta.agents.blocks.list(agent_id=agent_id)
                except AttributeError:
                    # Fallback if list is not directly supported this way
                    try:
                        agent_state = letta_client.letta.agents.get(agent_id=agent_id)
                        blocks = getattr(agent_state.memory, "blocks", [])
                    except AttributeError:
                        blocks = []
                
                current_labels = [n["id"] for n in get_all_nodes(dll_state)]
                recovered_count = 0
                for b in blocks:
                    label = b.get("label") if isinstance(b, dict) else getattr(b, "label", None)
                    content = b.get("value") if isinstance(b, dict) else getattr(b, "value", None)
                    
                    if not label:
                        continue
                        
                    if label not in current_labels and label not in letta_client.INITIAL_FIXED_BLOCKS:
                        try:
                            from memory.dll_manager import add_node
                            if "dynamic_block_count" in dll_state and dll_state["dynamic_block_count"] < dll_state["dynamic_block_max"]:
                                dll_node = {
                                    "id": label,
                                    "label": label.replace("_", " ").title(),
                                    "type": "projet",
                                    "keywords": [],
                                    "is_fixed": False,
                                    "active": True
                                }
                                # Use a safe add that doesn't trigger weaviate unless we want to, or use insert_node_by_type
                                # Actually we should just insert to DLL
                                from memory.block_factory import insert_node_by_type
                                dll_node["last_modified"] = datetime.now().isoformat()
                                dll_node["access_count"] = 0
                                dll_node["created_by"] = "recovered"
                                dll_state = insert_node_by_type("projet", dll_node, dll_state)
                                dll_state["dynamic_block_count"] += 1
                                st.session_state.memory_facts[label] = content
                                recovered_count += 1
                        except Exception as e:
                            st.warning(f"Failed to recover {label}: {e}")
                            
                if recovered_count > 0:
                    st.success(f"Recovered {recovered_count} block(s).")
                    save_dll(dll_state)
                    # We might need to run save_dll to disk
                    st.rerun()
                else:
                    st.warning("No missing blocks found in Letta.")
            except Exception as e:
                st.error(f"Failed to recover blocks: {e}")

# Initial sync of memory if empty
if not st.session_state.memory_facts:
    with st.spinner("Syncing Letta Memory..."):
        nodes = get_all_nodes(dll_state)
        ghost_blocks_deleted = False
        
        for node in nodes:
            b_id = node['id']
            content = get_core_block_content(agent_id, b_id)
            
            if content is None:  # Block returns 404 Not Found from Letta
                if not node.get("is_fixed", False):
                    # Automatic cleanup for orphaned dynamic blocks
                    try:
                        dll_state = delete_block_stitching(b_id, dll_state)
                        ghost_blocks_deleted = True
                    except Exception as e:
                        st.error(f"Cleanup error for {b_id}: {e}")
                else:
                    st.session_state.memory_facts[b_id] = ""
            else:
                st.session_state.memory_facts[b_id] = content
                
        if ghost_blocks_deleted:
            save_dll(dll_state)
            st.warning("🧹 Cleaned up orphaned dynamic blocks that were missing from Letta.")
            st.rerun()

# --- HEADER ---
c1, c2 = st.columns([4, 1])
with c1:
    st.title("✈️ Travel Agent — DLL Memory")
with c2:
    if st.button("🔄 Sync Cloud", use_container_width=True, help="Force reload from Letta Cloud"):
        with st.spinner("Forcing Cloud Sync..."):
            current_dll = load_dll()
            nodes = get_all_nodes(current_dll)
            fresh_facts = {}
            ghost_blocks_deleted = False
            for node in nodes:
                b_id = node["id"]
                content = get_core_block_content(agent_id, b_id)
                if content is None:
                    if not node.get("is_fixed", False):
                        try:
                            current_dll = delete_block_stitching(b_id, current_dll)
                            ghost_blocks_deleted = True
                        except Exception as e:
                            st.warning(f"Cleanup error for {b_id}: {e}")
                    fresh_facts[b_id] = ""
                else:
                    fresh_facts[b_id] = content
            st.session_state.memory_facts = fresh_facts
            st.session_state.sync_version += 1
            if ghost_blocks_deleted:
                save_dll(current_dll)
            st.success("Synced!")
            st.rerun()

m1, m2, m3 = st.columns(3)
m1.metric("Dynamic Blocks", f"{dll_state['dynamic_block_count']} / {dll_state['dynamic_block_max']}")
m2.metric("Fixed Blocks", "4")
m3.metric("Injected into LLM Context", f"{len(st.session_state.last_injected_ids)}")

st.divider()

# --- MAIN LAYOUT : TWO COLUMNS ---
col_left, col_right = st.columns([1, 1])

# ── LEFT COL: DLL STRUCTURE ──
with col_left:
    st.subheader("📚 DLL Structure — HEAD → TAIL")
    
    with st.expander("➕ Create a new dynamic block (manual)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            m_label = st.text_input("Label", placeholder="ex: Restaurants Porto", key="manual_label")
            m_type = st.selectbox("Type", ["temp", "projet", "fondamental"], key="manual_type")
        with col_b:
            m_keywords = st.text_input("Keywords (comma separated)", key="manual_kw")
            m_content = st.text_area("Initial content", key="manual_content")
            
        if st.button("🚀 Create Block", use_container_width=True):
            if m_label and m_content:
                m_id = m_label.lower().replace(" ", "_")
                kw_list = [k.strip() for k in m_keywords.split(",") if k.strip()]
                try:
                    dll_state = create_dynamic_block(
                        m_id, m_label, m_type, m_content, kw_list, "user_manual",
                        dll_state, letta_client, wcd_client
                    )
                    st.session_state.memory_facts[m_id] = m_content
                    st.success(f"Block '{m_label}' created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("### Current order: [HEAD] → [TAIL]")
    nodes = get_all_nodes(dll_state)
    
    v = st.session_state.sync_version

    for idx_pos, node in enumerate(nodes):
        b_id = node["id"]
        is_active = node.get("active", True)
        is_fixed = node.get("is_fixed", False)
        status_icon = "🟢" if is_active else "🔴"
        badge = "📌 Fixed" if is_fixed else "🔄 Dynamic"
        
        is_injected = b_id in st.session_state.last_injected_ids
        live_content = st.session_state.memory_facts.get(b_id, "")
        preview = f" — \"{live_content[:50]}...\"" if live_content else " (Empty)"
        
        with st.expander(f"{status_icon} [{idx_pos}] {node['label']} — {node['type']} | {badge}", expanded=is_injected):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Vectorized keywords:** " + ", ".join([f"`{k}`" for k in node.get("keywords", [])]))
                new_kw_str = st.text_input("Update keywords", value=", ".join(node.get("keywords", [])), key=f"kw_in_{b_id}_v{v}")
                if st.button("Re-vectorize", key=f"vec_btn_{b_id}_v{v}"):
                    new_kw_list = [k.strip() for k in new_kw_str.split(",") if k.strip()]
                    dll_state = update_node_keywords(b_id, new_kw_list, dll_state)
                    save_dll(dll_state)
                    st.success("Re-vectorized!")
                    st.rerun()
                
                curr_txt = st.text_area("Block content", value=live_content, height=150, key=f"txt_in_{b_id}_v{v}")
                if st.button("💾 Save (Sync all)", key=f"save_btn_{b_id}_v{v}", type="primary"):
                    try:
                        dll_state = update_block_content(
                            b_id, curr_txt, node.get("keywords", []), dll_state, letta_client, wcd_client
                        )
                        st.session_state.memory_facts[b_id] = curr_txt
                        st.success("Synced to Letta & Weaviate!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sync error: {e}")
            
            with c2:
                if is_fixed:
                    st.info("📌 Fixed Block")
                    st.caption("Always included in memory")
                else:
                    active_tog = st.checkbox("📌 Force Include (Override)", value=is_active, help="Forces the injection of this dynamic block to the AI.", key=f"act_tog_{b_id}_v{v}")
                    if active_tog != is_active:
                        toggle_block(b_id, active_tog, dll_state)
                        save_dll(dll_state)
                        st.rerun()
                
                if not is_fixed:
                    if st.button("🗑️ Delete", key=f"del_btn_{b_id}_v{v}", use_container_width=True):
                        dll_state = delete_block_stitching(b_id, dll_state)
                        save_dll(dll_state)
                        delete_block(agent_id, b_id)
                        st.rerun()

# ── RIGHT COL: CHAT ──
with col_right:
    st.subheader("💬 Travel Agent (LLM)")
    chat_container = st.container(height=650)
    
    with chat_container:
        # History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Proposals
        if st.session_state.pending_proposal:
            prop = st.session_state.pending_proposal
            with st.chat_message("assistant"):
                st.info(prop["proposal_message"])
                p1, p2 = st.columns(2)
                if p1.button(f"✅ Create '{prop['label']}'", key="create_prop"):
                    try:
                        create_dynamic_block(
                            prop["proposed_id"], prop["label"], prop["type"], 
                            prop["initial_content"], prop["keywords"], "agent_proposal",
                            dll_state, letta_client, wcd_client
                        )
                        st.session_state.memory_facts[prop["proposed_id"]] = prop["initial_content"]
                        st.session_state.pending_proposal = None
                        st.rerun()
                    except Exception as e: st.error(str(e))
                if p2.button("❌ Ignore", key="ignore_prop"):
                    st.session_state.pending_proposal = None
                    st.rerun()

# --- CHAT INPUT (Page Bottom) ---
if prompt := st.chat_input("Ask the architect your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.langchain_history.append(HumanMessage(content=prompt))
    
    # Rerender immediately to show user message
    st.rerun()

# --- ASYNC AGENT PROCESSING ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with col_right:
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Agent is traversing the DLL..."):
                    # Fresh graph each time to avoid loop issues
                    agent_app = create_dll_agent_graph()
                    inputs = {
                        "messages": st.session_state.langchain_history, 
                        "agent_id": agent_id,
                        "search_enabled": search_on,
                        "memory_only_mode": memory_only_on,
                        "strict_manual_mode": strict_manual_on
                    }
                    
                    import asyncio
                    result = asyncio.run(agent_app.ainvoke(inputs))
                    
                    new_messages = result.get("messages", [])
                    resp_text = new_messages[-1].content
                    
                    match = re.search(r"Memory: (.*?) \|", resp_text)
                    if match:
                        injected_str = match.group(1)
                        st.session_state.last_injected_ids = [s.strip() for s in injected_str.split("+") if s.strip() and s.strip() != "none"]
                    
                    st.session_state.messages.append({"role": "assistant", "content": resp_text})
                    st.session_state.langchain_history = new_messages
                    
                    # Set proposal if any
                    if result.get("needs_new_block") == "True":
                        st.session_state.pending_proposal = result.get("proposed_block_config")
                        
                    # Final sync of memory facts
                    dll_state = load_dll()  # Reload to get MTF changes
                    nodes_to_check = list(dll_state["nodes"].keys())
                    ghost_blocks_deleted = False
                    
                    for b_id in nodes_to_check:
                         content = get_core_block_content(agent_id, b_id)
                         if content is None:
                             if not dll_state["nodes"].get(b_id, {}).get("is_fixed", False):
                                 try:
                                     dll_state = delete_block_stitching(b_id, dll_state)
                                     ghost_blocks_deleted = True
                                 except:
                                     pass
                             else:
                                 st.session_state.memory_facts[b_id] = ""
                         else:
                             st.session_state.memory_facts[b_id] = content
                             
                    if ghost_blocks_deleted:
                        save_dll(dll_state)
                    
                    st.rerun()
