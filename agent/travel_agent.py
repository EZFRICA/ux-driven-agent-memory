import sys
import os
import asyncio

# Add root path for memory imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.dll_manager import search_memory, load_dll, save_dll, move_to_front, toggle_block
from memory.context_compiler import compile_working_context
from memory.letta_cloud_client import send_message
from agent.agent_graph_dll import _extract_and_update_memory
from logger import get_logger

logger = get_logger(__name__)

def handle_manage_memory(command: str, dll: dict) -> dict:
    parts = command.strip().split()
    if len(parts) == 1:
        # just /manage-memory
        print("\n--- DLL MEMORY STATE ---")
        nodes = dll["nodes"]
        current = dll["head_id"]
        while current:
            n = nodes[current]
            status = "ACTIVE" if n["active"] else "INACTIVE"
            print(f"[{current}] -> Type: {n['type']} | Status: {status} | Access: {n['access_count']}")
            current = n["next"]
        print("------------------------------")
        print("Commands: /manage-memory enable <id> | disable <id> | reset")
        return dll
        
    elif len(parts) == 3 and parts[1] == "enable":
        dll = toggle_block(parts[2], True, dll)
        save_dll(dll)
    elif len(parts) == 3 and parts[1] == "disable":
        dll = toggle_block(parts[2], False, dll)
        save_dll(dll)
    elif len(parts) == 2 and parts[1] == "reset":
        print("Reset not implemented. Delete metadata_links.json to re-init.")
        
    return dll


async def async_main():
    print("======================================================")
    print("   TRAVEL PLANNER AGENT (AMNESIA-PROOF MODE)  ")
    print("======================================================")
    print("Type '/manage-memory' to inspect blocks")
    print("Type 'quit' or 'exit' to stop")
    print("------------------------------------------------------\n")

    dll = await load_dll()
    agent_id = dll.get("agent_id")
    
    if not agent_id:
        print("[!] No Letta agent_id defined in metadata_links.json.")
        print("[!] Please run `python main.py` and select option 4 first.")
        return

    while True:
        try:
            user_input = input("\nYou > ")
            if not user_input.strip():
                continue
                
            if user_input.lower() in ["quit", "exit"]:
                break
                
            if user_input.startswith("/manage-memory"):
                dll = handle_manage_memory(user_input, dll)
                continue
                
            # 1. DLL Memory Pipeline (Vector Routing)
            print("\n[Processing DLL...]")
            relevant_blocks = await search_memory(user_input, dll)
            
            # 2. Context Compilation (Injecting relevant Letta blocks)
            working_context = await compile_working_context(agent_id, relevant_blocks, user_input)
            
            # 3. LLM Request via Letta/Gemini
            print("[Generating response...]")
            response = await send_message(agent_id, working_context, user_input)
            
            # Print response
            print("\nAgent  > " + response)
            
            # 4. MEMORY WRITE-BACK (The "Amnesia-Proof" Step)
            # This extracts new facts from the conversation and updates Letta + Weaviate
            print("[Extending memory...]")
            await _extract_and_update_memory(
                agent_id=agent_id,
                user_query=user_input,
                agent_response=response,
                relevant_blocks=relevant_blocks,
                dll=dll
            )
            
            # 5. Move-To-Front (Local state caching)
            if relevant_blocks:
                primary_block_id = relevant_blocks[0]['id']
                dll = move_to_front(primary_block_id, dll)
                save_dll(dll)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error] {e}")
            logger.exception("CLI Agent Pipeline error")

if __name__ == "__main__":
    asyncio.run(async_main())
