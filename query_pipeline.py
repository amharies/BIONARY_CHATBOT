import os
import json
import re
import sys
import google.generativeai as genai
from typing import Dict, Any
import textwrap

# --- Attempt to import Colab userdata module ---
try:
    from google.colab import userdata
    print("[Config] Running in Google Colab. Will use 'userdata' for API key.")
    _IN_COLAB = True
except ImportError:
    print("[Config] Not running in Google Colab. Will use 'os.environ' for API key.")
    _IN_COLAB = False

# --- This is the REAL retriever file ---
try:
    # This imports the retriever.py file you have in your Canvas
    import retriever as member3_retriever


except ImportError:
    print("="*50, file=sys.stderr)
    print("ERROR: Could not import 'retriever.py'.", file=sys.stderr)
    print("Make sure 'retriever.py' (BGE Hybrid Search version) is in the same directory.", file=sys.stderr)
    print("="*50, file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred importing retriever: {e}", file=sys.stderr); sys.exit(1)


# --- Configuration ---
try:
    API_KEY = None
    if _IN_COLAB:
        API_KEY = userdata.get('GEMINI_API_KEY')
    if API_KEY is None:
        print("[Config] Colab secret not found. Trying environment variable...")
        API_KEY = os.environ.get("GEMINI_API_KEY") # Use .get for safer check
    if API_KEY is None: raise ValueError("API key not found in Colab secrets or os.environ")
    genai.configure(api_key=API_KEY)
    print("[Config] Gemini API key configured successfully.")
except Exception as e:
    print("="*50, file=sys.stderr)
    print(f"ERROR: During configuration: {e}", file=sys.stderr)
    print("="*50, file=sys.stderr)
    sys.exit(1)

# --- END OF CONFIGURATION ---

try:
    generation_model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
except Exception as e:
    print(f"ERROR: Could not initialize Gemini model: {e}", file=sys.stderr); sys.exit(1)

# --- Main Pipeline Logic ---

def _parse_json_from_response(text: str) -> Dict[str, Any]:
    """Extracts a JSON object from a model's text response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match: print(f"Error: No JSON found in response: {text}"); return {"intent": "error", "query": "Could not parse."}
    json_str = match.group(0)
    try: return json.loads(json_str)
    except json.JSONDecodeError as e: print(f"Error: Invalid JSON: {e}\nRaw: {json_str}"); return {"intent": "error", "query": "Invalid JSON."}

def handle_user_query(user_question: str) -> str:
    """Main function to handle the end-to-end query pipeline."""
    current_year = 2025 # Assuming current year for context

    # --- 1. Parse the Query (1st Gemini Call) ---

    # --- *** FINAL UPDATED PROMPT *** ---
    # This prompt is aware of the new `club_name` and `event_domain` columns.
    parsing_prompt = textwrap.dedent(f"""
    You are a query-parsing agent for a university club's knowledge base.
    Your job is to convert a user's question into a JSON object, choosing the best tool.

    You have two tools:
    1. Relational DB (PostgreSQL) for structured facts (who, when, how many, list all).
    2. Vector DB (pgvector) for conceptual/descriptive questions (what, tell me about, topic search).

    --- DATABASE SCHEMA ---

    Table: 'events' (Structured Facts)
    Columns: [
        event_id (TEXT, PK, e.g., 'Circuit Craft'),
        name_of_event (TEXT),
        club_name (TEXT, e.g., 'BIONARY'),
        event_domain (TEXT, e.g., 'Hardware / IoT', 'AI / ML', 'Design / 3D Modeling', 'Hackathon / AI / Robotics'),
        date_of_event (DATE, format YYYY-MM-DD),
        time_of_event (TEXT),
        faculty_coordinators (TEXT),
        student_coordinators (TEXT),
        venue (TEXT),
        mode_of_event (TEXT, e.g., 'Offline', 'Online'),
        registration_fee (TEXT),
        speakers (TEXT),
        perks (TEXT)
    ]

    Table: 'chunks' (Semantic Search)
    Columns: [
        text_chunk (TEXT, e.g., 'A beginner friendly workshop...'),
        embedding (VECTOR)
    ]
    (Contains event descriptions, highlights, perks, club, and domain)

    --- JSON OUTPUT FORMAT ---
    {{"intent": "semantic", "query": "text to search for"}}
    OR
    {{"intent": "structured", "query": "SELECT ... FROM events WHERE ..."}}

    --- RULES ---

    1.  **Use SQL for Departments/Domains:** To find events by department (e.g., "robotics", "AI", "hardware"), you **MUST** query the `event_domain` column.
        (e.g., `event_domain ILIKE '%robotics%'`)

    2.  **Prioritize SQL for Facts:** You **MUST** use "intent: structured" for any other specific facts:
        - "Who" (e.g., 'speakers', 'faculty_coordinators')
        - "When" (e.g., 'date_of_event')
        - "How many" (e.g., `COUNT(event_id)`)
        - "List all" (e.g., `SELECT name_of_event FROM events`)

    3.  **Use Semantic Search:** Use "intent: semantic" ONLY for conceptual or descriptive questions:
        - "What is RAG?"
        - "Tell me about..."
        - "What did the Arduino workshop cover?"

    4.  **SQL Syntax:**
        - Use `ILIKE` for case-insensitive string matching.
        - Use `EXTRACT(YEAR FROM date_of_event)` to get the year.
        - Assume 'this year' is {current_year}, 'last year' is {current_year - 1}.

    ---

    User Question: "{user_question}"
    JSON Output:
    """)
    # --- *** END OF PROMPT UPDATE *** ---

    print(f"\n[Pipeline] Parsing query: '{user_question}'")
    try:
        parser_response = generation_model.generate_content(parsing_prompt)
        parsed_intent = _parse_json_from_response(parser_response.text)
    except Exception as e: print(f"Error during Gemini query parsing: {e}"); return "Sorry, I had trouble understanding."
    print(f"[Pipeline] Parsed Intent: {parsed_intent}")

    # --- 2. Retrieve Data ---
    context = ""
    intent = parsed_intent.get("intent")
    query = parsed_intent.get("query")
    # This variable will hold the SQL query to pass to the final prompt for context
    sql_query_for_prompt = None

    if intent == "semantic":
        if not query: context = "Parser error: Missing query text for semantic search."
        # Call the hybrid search function
        else: context_docs = member3_retriever.query_vector_db(query); context = "\n".join(context_docs)
    elif intent == "structured":
        if not query: context = "Parser error: Missing SQL query for structured search."
        else:
            sql_query_for_prompt = query # <-- Store the query
            sql_results = member3_retriever.query_relational_db(query)
            context = f"Database query returned: {sql_results}"
    else: context = f"Query parser failed or returned unknown intent: {intent}"
    print(f"[Pipeline] Retrieved Context: {context[:300]}...") # Print longer snippet

    # --- 3. Generate Final Response (2nd Gemini Call) ---

    # --- *** FINAL PROMPT FIX *** ---
    # This prompt is now "SQL-aware" and "helpful"
    final_prompt = textwrap.dedent(f"""
    You are the 'Club Knowledge Search Agent'. Your job is to synthesize a final answer
    from the provided context. You MUST answer the user's question.

    You are given the user's question, the retrieved context, and (if applicable)
    the SQL query that was run to get that context.

    ---
    User Question:
    {user_question}
    ---
    Context:
    {context}
    ---
    SQL Query (if any):
    {sql_query_for_prompt if sql_query_for_prompt else 'N/A'}
    ---

    INSTRUCTIONS:
    1.  **If the Context is a Database Result (e.g., `Database query returned: [(1,)]`):**
        * Look at the 'SQL Query' and the 'Context' *together*.
        * If the query was `SELECT COUNT...` and the result is `[(1,)]`, the answer is 1.
        * If the query was `SELECT COUNT...` and the result is `[(0,)]`, the answer is 0.
        * Synthesize the raw SQL result into a natural, human-readable sentence.
        * **Example:** If query was `SELECT COUNT...` and result is `[(1,)]` and question was "How many robotics events...", answer "There was 1 robotics event."
        * **Example:** If query was `SELECT speakers...` and result is `[('Dr. A',), ('Dr. B',)]`, answer "The speakers were Dr. A and Dr. B."
        * If the result is `[('No results found for that query.',)]` or `[]` or `[('',)]`, state "I do not have that information in my records."

    2.  **If the Context is Semantic Text (from vector search):**
        * Read the text chunks to find the answer.
        * **Be helpful:** If the user asks about a specific topic (e.g., "RAG"), and the context shows an event *mentions* that topic (even in 'perks' or 'highlights'), you **MUST** state that you found a mention and present the details (e.g., "Yes, the 'From Voice to Notes' event mentioned RAG in its perks...").
        * If the answer is not in the text, state "I do not have that information in my records."

    3.  **Do not make up information.** Answer ONLY from the context.

    Final Answer:
    """)
    # --- *** END OF PROMPT FIX *** ---

    print("[Pipeline] Generating final answer...")
    try:
        final_response = generation_model.generate_content(final_prompt)
        if final_response.prompt_feedback.block_reason:
             return f"Sorry, the response was blocked. Reason: {final_response.prompt_feedback.block_reason}"
        return final_response.text
    except Exception as e: print(f"Error during Gemini final response: {e}"); return "Sorry, I had trouble formulating a response."

# --- 4. Interactive Chat Loop ---
if __name__ == "__main__":
    print("\n--- Club Knowledge Search Agent ---")
    print("Ask me questions about past club events. Type 'quit' or 'exit' to end.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Agent: Goodbye!")
                break
            if not user_input.strip(): continue

            answer = handle_user_query(user_input)
            print(f"Agent: {answer}")

        except KeyboardInterrupt:
            print("\nAgent: Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")



