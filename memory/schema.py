"""
Weaviate Schema — defines and initializes all WCD collections for the DLL architecture.

Collections:
    - TravelFixed:  Full content of fixed memory blocks (profile, preferences, trip, session)
    - TravelDynamic: Full content of dynamic memory blocks (created during conversation)
    - BlockIndex:   Lightweight keyword vectors for DLL routing (BMJ algorithm)
"""

import os
import sys

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weaviate.classes.config import Property, DataType, Tokenization, Configure
from weaviate.classes.init import Auth
import weaviate
from config import WCD_CLUSTER_URL, WCD_API_KEY, GEMINI_API_KEY
from logger import get_logger

logger = get_logger(__name__)


def get_weaviate_client():
    """Connect to the Weaviate Cloud cluster (Sync)."""
    if not WCD_CLUSTER_URL or not WCD_API_KEY:
        raise ValueError("Missing Weaviate environment variables.")

    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WCD_CLUSTER_URL,
        auth_credentials=Auth.api_key(WCD_API_KEY),
        headers={"X-Goog-Api-Key": GEMINI_API_KEY},
        skip_init_checks=True,
    )

def get_weaviate_client_async():
    """Connect to the Weaviate Cloud cluster (Async)."""
    if not WCD_CLUSTER_URL or not WCD_API_KEY:
        raise ValueError("Missing Weaviate environment variables.")

    return weaviate.use_async_with_weaviate_cloud(
        cluster_url=WCD_CLUSTER_URL,
        auth_credentials=Auth.api_key(WCD_API_KEY),
        headers={"X-Goog-Api-Key": GEMINI_API_KEY},
        skip_init_checks=True,
    )


def init_block_index_schema():
    """
    Initialize the BlockIndex collection — used for DLL routing (BMJ algorithm).
    The 'keywords_text' field is vectorized by Weaviate for near_text search.
    """
    client = get_weaviate_client()
    try:
        if not client.collections.exists("BlockIndex"):
            client.collections.create(
                name="BlockIndex",
                description="Keyword metadata vectors for DLL routing (BMJ algorithm)",
                multi_tenancy_config=Configure.multi_tenancy(enabled=True),
                properties=[
                    Property(
                        name="keywords_text",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.WORD,
                        description="Block keywords joined as text — auto-vectorized for near_text routing",
                    ),
                    Property(
                        name="block_id",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="Unique block identifier (e.g. 'traveler_preferences')",
                    ),
                    Property(
                        name="agent_id",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="Letta Agent ID for isolation",
                    ),
                    Property(
                        name="block_type",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="Semantic type: fondamental, projet, or temp",
                    ),
                    Property(
                        name="is_active",
                        data_type=DataType.BOOL,
                        description="Whether the block is active in LLM context",
                    ),
                ],
                vector_config=Configure.Vectors.text2vec_google_gemini(
                    name="keywords_text",
                    model="gemini-embedding-2-preview",
                ),
            )
            logger.info("Created collection: BlockIndex")
        else:
            logger.debug("Collection BlockIndex already exists.")
    finally:
        client.close()


def init_travel_fixed_schema():
    """
    Initialize the TravelFixed collection — stores full content of fixed memory blocks.
    Used for archival and content retrieval (not for routing).
    """
    client = get_weaviate_client()
    try:
        if not client.collections.exists("TravelFixed"):
            client.collections.create(
                name="TravelFixed",
                description="Fixed memory blocks — profile, preferences, trip, session",
                multi_tenancy_config=Configure.multi_tenancy(enabled=True),
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.WORD,
                        description="Full block content text",
                    ),
                    Property(
                        name="block_id",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="Unique block identifier",
                    ),
                    Property(
                        name="block_type",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="Semantic type: fondamental, projet, or temp",
                    ),
                    Property(
                        name="tags",
                        data_type=DataType.TEXT_ARRAY,
                        description="Metadata tags for filtering",
                    ),
                    Property(
                        name="updated_at",
                        data_type=DataType.DATE,
                        description="Last content update timestamp",
                    ),
                ],
                vector_config=Configure.Vectors.text2vec_google_gemini(
                    name="content",
                    model="gemini-embedding-2-preview",
                ),
            )
            logger.info("Created collection: TravelFixed")
        else:
            logger.debug("Collection TravelFixed already exists.")
    finally:
        client.close()


def init_travel_dynamic_schema():
    """
    Initialize the TravelDynamic collection — stores full content of dynamic blocks.
    Created during conversation (e.g. 'porto_itinerary', 'budget_tracker').
    """
    client = get_weaviate_client()
    try:
        if not client.collections.exists("TravelDynamic"):
            client.collections.create(
                name="TravelDynamic",
                description="Dynamic memory blocks created during conversation",
                multi_tenancy_config=Configure.multi_tenancy(enabled=True),
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.WORD,
                        description="Full block content text",
                    ),
                    Property(
                        name="block_id",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="Unique block identifier",
                    ),
                    Property(
                        name="block_type",
                        data_type=DataType.TEXT,
                        tokenization=Tokenization.FIELD,
                        description="Semantic type: fondamental, projet, or temp",
                    ),
                    Property(
                        name="tags",
                        data_type=DataType.TEXT_ARRAY,
                        description="Metadata tags for filtering",
                    ),
                    Property(
                        name="created_at",
                        data_type=DataType.DATE,
                        description="Block creation timestamp",
                    ),
                    Property(
                        name="updated_at",
                        data_type=DataType.DATE,
                        description="Last content update timestamp",
                    ),
                ],
                vector_config=Configure.Vectors.text2vec_google_gemini(
                    name="content",
                    model="gemini-embedding-2-preview",
                ),
            )
            logger.info("Created collection: TravelDynamic")
        else:
            logger.debug("Collection TravelDynamic already exists.")
    finally:
        client.close()


def init_all_schemas():
    """Initialize all 3 Weaviate collections for the DLL architecture."""
    init_block_index_schema()
    init_travel_fixed_schema()
    init_travel_dynamic_schema()
    logger.info("All Weaviate schemas initialized.")


if __name__ == "__main__":
    init_all_schemas()
