%%writefile retriever.py
import os
import sys
import psycopg2
import google.colab.userdata
from typing import List, Tuple, Set
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Number of results to fetch from each search type
TOP_K = 5 # Fetch 5 from vector, 5 from keyword

# 1. Load the model.
try:
    print(f"[Retriever] Loading sentence-transformer model '{MODEL_NAME}'...")
    # Add trust_remote_code=True if needed by the specific model version
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Failed loading with trust_remote_code=True, trying without: {e}", file=sys.stderr)
        model = SentenceTransformer(MODEL_NAME)

    # Verify model dimension
    actual_dimension = model.get_sentence_embedding_dimension()
    if actual_dimension != 768:
        print(f"--- FATAL ERROR --- \nModel '{MODEL_NAME}' loaded with wrong dimension!", file=sys.stderr)
        print(f"Expected 768, but got {actual_dimension}", file=sys.stderr)
        model = None
    else:
        print(f"[Retriever] Model loaded successfully (768 dimensions).")

except Exception as e:
    print(f"ERROR: Could not load sentence-transformer model: {e}", file=sys.stderr)
    model = None

def _connect_to_db():
    """Connects to the Neon DB using the Colab secret."""
    try:
        conn_string = google.colab.userdata.get('NEON_DB_URL')
        if conn_string is None:
            print("ERROR: [Retriever] 'NEON_DB_URL' secret not found.", file=sys.stderr)
            return None
        conn = psycopg2.connect(conn_string)
        return conn
    except Exception as e:
        print(f"ERROR: [Retriever] Unable to connect to database: {e}", file=sys.stderr)
        return None

# --- Relational DB Query ---
def query_relational_db(sql_query: str) -> List[Tuple]:
    """ Executes a read-only SQL query against the Neon (PostgreSQL) database. """
    print(f"[Retriever] Received SQL query: '{sql_query}'")
    conn = _connect_to_db()
    if conn is None: return [("Database connection error.",)]
    results = []
    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            if cur.description: results = cur.fetchall()
            else: results = []
    except (psycopg2.Error, Exception) as e:
        print(f"ERROR: [Retriever] SQL query failed: {e}", file=sys.stderr)
        conn.rollback(); results = [(f"SQL error: {e}",)]
    finally:
        if conn: conn.close()
    if not results: return [("No results found for that query.",)]
    return results

# --- Vector DB Query (Hybrid Search) ---
def query_vector_db(query_text: str) -> List[str]:
    """
    Performs HYBRID search (Vector + Keyword) against the Neon database.
    """
    if model is None: return ["Error: SentenceTransformer model is not loaded."]

    print(f"[Retriever] Received SEMANTIC query for: '{query_text}'")
    conn = _connect_to_db()
    if conn is None: return ["Database connection error."]

    # Use a dictionary to store results and their ranks for re-ranking
    # { text_chunk: rank } - lower rank is better
    combined_results_map = {}

    # --- 1. Vector Search ---
    print("[Retriever] Performing vector search...")
    try:
        query_with_prefix = QUERY_PREFIX + query_text
        query_embedding = model.encode(query_with_prefix)
        with conn.cursor() as cur:
            register_vector(cur)
            cur.execute(
                f"""
                SELECT text_chunk, embedding <-> %s AS distance
                FROM chunks
                ORDER BY distance
                LIMIT {TOP_K};
                """,
                (query_embedding,)
            )
            rows = cur.fetchall()
            for i, row in enumerate(rows):
                combined_results_map[row[0]] = i # Rank 0, 1, 2, 3, 4
            print(f"[Retriever] Vector search found {len(rows)} results.")

    except (psycopg2.Error, Exception) as e:
        print(f"ERROR: [Retriever] Vector query failed: {e}", file=sys.stderr)
        conn.rollback()

    # --- 2. Keyword Search (Full-Text Search) ---
    print("[Retriever] Performing keyword search...")
    try:
        if conn.closed: # Reconnect if vector search failed and rolled back
             conn = _connect_to_db()
             if conn is None: raise Exception("Reconnect failed for keyword search")

        with conn.cursor() as cur:
            # Use `websearch_to_tsquery` - it's better for user queries and acronyms like "RAG"
            keyword_query = """
                SELECT text_chunk, ts_rank_cd(to_tsvector('english', text_chunk), query) AS rank
                FROM chunks, websearch_to_tsquery('english', %s) query
                WHERE query @@ to_tsvector('english', text_chunk)
                ORDER BY rank DESC
                LIMIT {TOP_K};
            """
            cur.execute(keyword_query, (query_text,))
            rows = cur.fetchall()
            for row in rows:
                if row[0] not in combined_results_map:
                    # Add keyword-only results with a lower priority (higher rank number)
                    combined_results_map[row[0]] = 100
            print(f"[Retriever] Keyword search found {len(rows)} results.")

    except (psycopg2.Error, Exception) as e:
        print(f"ERROR: [Retriever] Keyword query failed: {e}", file=sys.stderr)

    finally:
        if conn and not conn.closed:
            conn.close()

    # --- 3. Combine and Rank Results ---
    # Sort by rank (lower is better), which prioritizes vector results
    ranked_results = sorted(combined_results_map.keys(), key=lambda x: combined_results_map[x])

    print(f"[Retriever] Combined search yields {len(ranked_results)} unique results.")

    if not ranked_results:
        return ["No relevant documents found."]

    # Return the top 10 (or fewer) combined unique results

    return ranked_results[:10]
