# UX-Driven Agent Memory 🧠✈️

This project explores a novel architecture for building dynamic, user-transparent memory for AI agents (focusing on a Travel Planner Use-Case). It replaces the traditional "black box" RAG pipeline with a **DLL (Doubly Linked List)** architecture powered by the **BMJ (Bidirectional Metadata Jump)** algorithm, backed by Weaviate and Letta Cloud.

## 🚀 Key Features

*   **Transparent Memory (DLL)**: Memory is structured as an ordered list of semantic blocks (Priority HEAD → TAIL).
*   **User Override & Control**: Users can see exactly what the AI agent remembers and can *force-include* or manually edit blocks via the Streamlit Dashboard.
*   **BMJ Algorithm**: A vector-assisted routing algorithm that selects reading directions and fills context gaps dynamically without overflowing the LLM.
*   **Letta Core Memory**: Factual write-back to Letta Cloud for long-term state persistence.
*   **Weaviate BlockIndex**: Lightweight keyword vectorization for high-speed block routing (powered by Gemini embeddings).

## 🛠️ Architecture

1.  **Dashboard (`dashboard/app.py`)**: A Streamlit interface where users can interact with the agent *and* control its memory DLL simultaneously.
2.  **DLL Manager (`memory/dll_manager.py`)**: The brain of the memory system utilizing the BMJ algorithm.
3.  **Letta Client (`memory/letta_cloud_client.py`)**: Syncs with Letta to maintain "Core Memory" blocks.
4.  **Weaviate Client (`memory/weaviate_cloud_client.py`)**: Maintains the metadata BlockIndex for smart context retrieval.
5.  **Agent Logic (`agent/agent_graph_dll.py`)**: A LangGraph state-graph that handles planning, API tools (Google Search Grounding), and automatic memory write-back (Fact extraction).

## 📦 Setup & Installation

### 1. Prerequisites
You will need Python 3.10+ (using `uv` is recommended) and the following API keys:
*   [Google Gemini API Key](https://aistudio.google.com/)
*   [Letta Cloud API Key](https://letta.com/)
*   [Weaviate Cloud (WCD) Cluster URL & API Key](https://console.weaviate.cloud/)

### 2. Environment Variables
Clone the repository and copy the example environment file:
```bash
cp .env.example .env
```
Fill in the `.env` file with your actual keys:
```env
GEMINI_API_KEY=your_gemini_key
LETTA_API_KEY=your_letta_key
LETTA_BASE_URL=https://api.letta.com
WCD_CLUSTER_URL=your_weaviate_cluster_url
WCD_API_KEY=your_weaviate_api_key
USER_ID=user_abc123
```

### 3. Install Dependencies
Using `uv`:
```bash
uv sync
```

### 4. Initialization
Before launching the UI, you need to create the required Weaviate schemas, create the Letta agent, and synchronize the locally generated metadata with Weaviate.

1.  **Init Weaviate Schemas:**
    ```bash
    uv run memory/schema.py
    ```
2.  **Create the Letta Agent:**
    ```bash
    uv run memory/letta_cloud_client.py --create-agent
    ```
    *This will auto-save the generated `agent_id` into your local `metadata_links.json`.*

3.  **Synchronize Default Blocks to Weaviate:**
    ```bash
    uv run memory/sync_memory.py
    ```
    *This indexes the 4 foundational blocks (Profile, Preferences, etc.) into the Weaviate BlockIndex, associating them strictly with your new `agent_id`.*

### 5. Running the Application
Launch the rich Streamlit dashboard:
```bash
uv run streamlit run dashboard/app.py
```
*(Optionally run head-less CLI agent: `python agent/travel_agent.py`)*

## 🧠 Theory of Operation

The system defines an absolute max context limit of **12 blocks**. 
*   **4 Fixed Blocks** (Profile, Preferences, Active Trip, Session) are *always* included to frame the identity of the agent.
*   **User Checkboxes ("Force Override")**: If a user checks a dynamic block in the interface, it bypasses the algorithm and is *injected* straight into the AI's prompt.
*   **BMJ Fill**: Unchecked blocks are semantically scored against the user's latest query. High scoring blocks are automatically grabbed by the algorithm to fill out the remaining available context slots.

## 🧪 Demo & Test Scenarios

### 1. Cold Start & Onboarding (Implicit Knowledge)
Prove that the agent understands the user's profile without explicit mentions:
1. In the sidebar, manually edit **Traveler Profile**:
   - Keywords: `profile, identity, passport, name, client`
   - Content: *"My name is Paul, 35 years old. My French passport (FR98765) is valid until 2029."*
2. Edit **Traveler Preferences**:
   - Keywords: `preferences, stay, budget, food, flight`
   - Content: *"Solo traveler. I love Japanese culture, local street-food, and I enjoy staying in small traditional hotels (Ryokans). Moderate budget."*
3. **The Test:** In the Chat, ask: *"If I were to go to Asia, what kind of accommodation would you spontaneously recommend?"*
4. **Result:** Thanks to the BMJ algorithm, the agent automatically loads your preferences and suggests **Ryokans** or traditional guesthouses, proving it knows who Paul is and what he likes without him stating it in the query.

### 2. Selective Retrieval (The "Secret Info" Test)
Prove that the agent only knows what is in the DLL:
1. Edit or Create a dynamic block named **"Tokyo Restaurants"**.
2. Content: *"For grilled food, absolutely go to 'Kenta-San' in the hidden alley behind Hanazono Temple. It's owned by my friend Tanaka's cousin. Tell him you come from Paul to get the free sake."*
3. Keywords: `tanaka, kenta, grilled, secret, cousin`.
4. **Test (OFF):** Uncheck "Force Include" and ask: *"I want to go to Tanaka's cousin's place tonight, do you have the address?"*. The agent finds it via BMJ.
5. **Test (STRICT):** Turn on **"Disable BMJ Auto-Retrieval"** (Demo Mode) in the sidebar. Ask again. The agent will now be "amnesic" about Tanaka's cousin.

### 3. Automatic Memory Write-back
Observe the agent taking notes:
1. Message the agent: *"My new passport number is FR777 and I'm allergic to peanuts."*
2. **Result:** After the response, refresh the page. Look at the `Traveler Profile` and `Traveler Preferences` blocks. The info has been automatically extracted and persisted to Letta Cloud.

### 3. Real-time Monitoring (Visualizer)
See the "Brain" move:
1. Open a new terminal: `uv run python visualizer/server.py`.
2. Go to `http://localhost:8080`.
3. Ask about your profile or a specific dynamic block. Watch the nodes animatedly jump to the **HEAD** position as the Move-To-Front (MTF) algorithm prioritizes the most useful context.

### 🚀 Production & Scaling (2M+ Users)

The current architecture is optimized for high-concurrency (Async) and data isolation (Weaviate Multi-tenancy). To support **2,000,000+ users** in a distributed environment, the local JSON storage for DLL states should be replaced by a distributed store like **Redis**.

### Why Redis for DLL?
*   **Speed**: Sub-millisecond latency for memory retrieval (BMJ routing).
*   **Concurrency**: Atomic operations prevent state corruption across multiple API workers.
*   **Scalability**: Allows horizontal scaling of your FastAPI/Python backend.

### Recommended Redis Implementation

For production, replace `load_dll()` and `save_dll()` in `memory/dll_manager.py` with an async Redis implementation:

```python
import redis.asyncio as redis
import json

REDIS_URL = "redis://localhost:6379/0"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

async def save_dll_redis(user_id: str, dll: dict):
    """Persist DLL state to Redis with TTL (e.g., 30 days)."""
    key = f"agent_memory:dll:{user_id}"
    await redis_client.set(key, json.dumps(dll), ex=2592000)

async def load_dll_redis(user_id: str) -> dict:
    """Fetch DLL state from Redis. Fallback to init_dll if missing."""
    key = f"agent_memory:dll:{user_id}"
    data = await redis_client.get(key)
    if data:
        return json.loads(data)
    return await init_dll_async(user_id)
```

---

## 🛠️ Architecture Recap (Robustness)

| Component | Dev/Test (Local) | Production (Scaling) |
|---|---|---|
| **Memory Extraction** | Letta Cloud (Async) | Letta Self-Hosted (K8s) |
| **Vector Index** | WCD (Tenant Isolation) | Weaviate Multi-tenancy |
| **State Storage** | **Atomic JSON Files** | **Redis / PostgreSQL** |
| **Orchestration** | Python / LangGraph | FastAPI / Pure Async |
| **Concurrency** | Threadpool Fallback | Native Async IO |

### 4. Multi-Tenant Isolation
Ensure agent-level data security:
1. Note your current `agent_id` in the sidebar.
2. Run `uv run memory/letta_cloud_client.py --create-agent` to generate a NEW agent.
3. Restart Streamlit.
4. **Result:** All memories from the previous agent are invisible. The Weaviate search is strictly filtered by `agent_id`.