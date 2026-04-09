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