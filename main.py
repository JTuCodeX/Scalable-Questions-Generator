
# ======================
# 1. Setup & Imports
# ======================

# --- Standard Libraries ---
import os
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed  # For batching + parallel API calls

# --- PDF/Text Processing ---
from pypdf import PdfReader  # Extracts raw text from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into chunks

# --- Embeddings + Vector Store ---
from langchain_huggingface import HuggingFaceEmbeddings  # Generates embeddings using HuggingFace models
from langchain_community.vectorstores import FAISS       # FAISS: Fast Approximate Nearest Neighbors search

# --- Google Gemini API ---
from google import genai
from google.genai import types

# ======================
# 2. API Client Setup
# ======================


# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Choose model version (fast + cost-efficient)
MODEL_NAME = "gemini-2.0-flash-lite"



# ======================
# 2. Load Documents (PDF / TXT)
# ======================
def load_document(path: str) -> str:
    """
    Loads text from a given document.

    Supported formats:
      - PDF (.pdf): Extracts text from each page.
      - TXT (.txt): Reads the entire text file.

    Args:
        path (str): Path to the input document (PDF or TXT).

    Returns:
        str: Extracted text content.

    Raises:
        ValueError: If the file extension is not supported.
    """

    if path.endswith(".pdf"):
        # Read PDF and extract text page by page
        reader = PdfReader(path)
        text = " ".join([page.extract_text() or "" for page in reader.pages])

    elif path.endswith(".txt"):
        # Read plain text file
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    else:
        raise ValueError("Unsupported file format. Please use PDF or TXT.")

    return text

# ======================
# 3. Adaptive Chunking
# ======================

def chunk_text(text, target_chunks=50, overlap=100):
    """
    Splits a large text into smaller chunks adaptively.

    Parameters
    text : str
        The raw extracted text (from PDF or TXT).
    target_chunks : int, default=50
        The approximate number of chunks we want to split the text into.
        (Helps balance between too many small chunks vs. too few large ones.)
    overlap : int, default=100
        Number of overlapping characters between consecutive chunks.
        This preserves context across chunk boundaries.

    Returns
    list of str
        A list of text chunks ready for embeddings / retrieval.
    """

    # Total length of the document in characters
    total_chars = len(text)

    # Dynamically calculate chunk size based on total length and target number of chunks
    # Ensures longer documents have larger chunks, while shorter ones still split properly
    chunk_size = max(500, math.ceil(total_chars / target_chunks))

    # Use LangChain's RecursiveCharacterTextSplitter to split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # computed adaptive size
        chunk_overlap=overlap   # overlap to retain continuity
    )

    # Return the final list of chunks
    return splitter.split_text(text)

# ======================
# 4. Batched Main Idea Extraction
# ======================

def extract_main_ideas_batch(chunks, ideas_per_chunk=2, batch_size=5, budget=None):
    """
    Extracts high-level main ideas from text chunks in batches.

    Args:
        chunks (list[str]): List of text chunks from the document.
        ideas_per_chunk (int): Target number of ideas to extract per chunk (not enforced strictly).
        batch_size (int): Number of chunks to process per API request.
        budget (dict, optional): A mutable dictionary with {"calls": int}
                                 that tracks remaining API calls for quota management.

    Returns:
        list[str]: A flat list of extracted main ideas.
    """

    main_ideas = []

    # Process chunks in batches to reduce API calls
    for i in range(0, len(chunks), batch_size):

        # Stop early if API budget is exhausted
        if budget and budget["calls"] <= 0:
            break

        # Select a batch of text chunks
        batch = chunks[i:i + batch_size]

        # Construct prompt for Gemini to extract ideas
        prompt = "Extract the main idea(s) from each of the following passages:\n\n"
        for idx, text in enumerate(batch, start=1):
            prompt += f"Passage {idx}: {text}\n\n"

        try:
            # Call Gemini API with batched prompt
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            # Parse response: split into lines and clean bullets/dashes
            ideas = [
                line.strip("-• ")
                for line in res.text.split("\n")
                if line.strip()
            ]

            # Collect all extracted ideas
            main_ideas.extend(ideas)

            # Deduct from budget if applicable
            if budget:
                budget["calls"] -= 1

        except Exception as e:
            # Fail gracefully instead of crashing pipeline
            print(f"Error during batched main idea extraction: {e}")

    return main_ideas

# ======================
# 5. Build Index & Retrieve Passages (Batched)
# ======================

def build_index(chunks):
    """
    Builds a FAISS vector index from the given text chunks.

    Args:
        chunks (list[str]): A list of text segments extracted from the document.

    Returns:
        FAISS: A FAISS index containing embeddings of all text chunks.
    """
    # Load a lightweight, efficient embedding model for semantic similarity.
    # The "all-MiniLM-L6-v2" model balances speed and accuracy well.
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS index from the chunks using the embeddings.
    return FAISS.from_texts(chunks, embedder)


def retrieve_passages_batch(main_ideas, index, k=3, batch_size=5):
    """
    Retrieves the top-k most relevant passages for each main idea in batches.

    Args:
        main_ideas (list[str]): Extracted main concepts or questions to search for.
        index (FAISS): The FAISS index built from document chunks.
        k (int, optional): Number of top passages to retrieve per idea. Default = 3.
        batch_size (int, optional): Number of ideas to process in one batch. Default = 5.

    Returns:
        list[dict]: A list of dictionaries where each entry contains:
            {
                "idea": str,           # The input main idea
                "passages": list[str]  # Retrieved passages relevant to that idea
            }
    """
    results = []

    # Process main ideas in batches to avoid excessive API calls / memory usage
    for i in range(0, len(main_ideas), batch_size):
        batch = main_ideas[i:i+batch_size]   # Select a batch of main ideas
        batch_results = []

        # For each idea, retrieve top-k most similar passages from FAISS index
        for idea in batch:
            docs = index.similarity_search(idea, k=k)
            batch_results.append({
                "idea": idea,
                "passages": [doc.page_content for doc in docs]
            })

        # Add the batch results to the global list
        results.extend(batch_results)

    return results

# ======================
# 6. Batched MCQ Generation (Savaal-Inspired Prompt)
# ======================

def generate_mcqs_batch(idea_passage_pairs, questions_per_idea=2, batch_size=2, budget=None):
    """
    Generate multiple-choice questions (MCQs) in batches from extracted ideas + supporting passages.

    Parameters
    ----------
    idea_passage_pairs : list of dict
        A list where each item is of the form:
        {
            "idea": str,          # main extracted concept
            "passages": [str...]  # supporting text passages retrieved via FAISS
        }

    questions_per_idea : int, optional (default=2)
        Number of MCQs to generate for each idea.

    batch_size : int, optional (default=2)
        Number of idea-passage pairs to send to the LLM per request.
        Larger batch sizes reduce API calls but increase prompt size.

    budget : dict, optional
        A mutable dict for quota control, e.g. {"calls": 10}.
        If provided, the function decrements `budget["calls"]` after each API request
        and stops when quota is exhausted.

    Returns
    -------
    all_questions : dict
        A dictionary grouped by difficulty level:
        {
            "easy":   [ {question, choices, answer}, ... ],
            "medium": [ {question, choices, answer}, ... ],
            "hard":   [ {question, choices, answer}, ... ]
        }
    """

    # Initialize container for all questions grouped by difficulty
    all_questions = {"easy": [], "medium": [], "hard": []}

    # Process in batches (for scalability + quota efficiency)
    for i in range(0, len(idea_passage_pairs), batch_size):

        # Stop if budget is depleted
        if budget and budget["calls"] <= 0:
            break

        batch = idea_passage_pairs[i:i+batch_size]

        # Construct system prompt for Gemini
        # (inspired by the Savaal pipeline, with strict JSON schema)
        prompt = f"""
        You are an expert exam question generator.
        For each main idea and passage pair below, generate {questions_per_idea} multiple-choice questions.
        The questions should test deep understanding, critical thinking, and analysis.

        Requirements:
        - Avoid trivial or duplicate questions.
        - Do not use the phrases "main idea" or "passages" in the question text.
        - Use Bloom’s taxonomy for difficulty:
            * Easy   - remembering facts, definitions.
            * Medium - understanding/explaining concepts.
            * Hard   - applying, analyzing, evaluating.

        Output strictly in JSON with this schema:
        {{
            "easy": [
              {{
                "question": "string",
                "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
                "answer": "string"
              }}
            ],
            "medium": [...],
            "hard": [...]
        }}
        """

        # Append batch-specific content to the prompt
        for idx, pair in enumerate(batch, start=1):
            idea = pair["idea"]
            passages = "\n".join(pair["passages"])
            prompt += f"\nMain Idea {idx}: {idea}\nPassages {idx}: {passages}\n"

        try:
            #  Call Gemini model
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            if budget:  # decrement quota
                budget["calls"] -= 1

            try:
                # Parse JSON response
                questions = json.loads(res.text)

                if isinstance(questions, dict):
                    #  Expected schema: dict with keys "easy", "medium", "hard"
                    for level in ["easy", "medium", "hard"]:
                        all_questions[level].extend(questions.get(level, []))

                elif isinstance(questions, list):
                    #  Fallback: if model outputs a raw list instead of dict
                    print("Model returned a list instead of dict, defaulting to 'medium'")
                    all_questions["medium"].extend(questions)

                else:
                    print("Unexpected JSON type, skipping batch")

            except json.JSONDecodeError:
                print("Failed to parse JSON, skipping batch")
                continue

        except Exception as e:
            # Catch API errors or connection issues
            print(f" Error during batched question generation: {e}")
            continue

    return all_questions

def filter_grouped_questions(grouped):
    """
    Filters and validates grouped multiple-choice questions.

    Args:
        grouped (dict): Dictionary of questions grouped by difficulty.
                        Expected structure:
                        {
                          "easy": [ { "question": ..., "choices": [...], "answer": ... }, ... ],
                          "medium": [...],
                          "hard": [...]
                        }

    Returns:
        dict: Cleaned questions grouped by difficulty, with invalid/duplicate ones removed.
    """

    # Initialize the output structure with empty lists
    valid = {"easy": [], "medium": [], "hard": []}

    # Iterate through each difficulty level
    for level in ["easy", "medium", "hard"]:
        questions = grouped.get(level, [])  # Retrieve questions for this level
        seen = set()  # Track duplicates by normalized question text

        for q in questions:
            # Normalize the question text for duplicate checking
            q_text = q.get("question", "").strip().lower()

            # -------------------------
            # Schema validation
            # -------------------------
            # Must have exactly 4 answer choices
            if len(q.get("choices", [])) != 4:
                continue
            # The correct answer must appear among the choices
            if q.get("answer") not in q["choices"]:
                continue

            # -------------------------
            #  Quality filtering
            # -------------------------
            # Skip duplicates (case-insensitive)
            if q_text in seen:
                continue
            # Skip very short/low-quality questions (< 5 words)
            if len(q.get("question", "").split()) < 5:
                continue

            # If it passes all filters, add it to the valid set
            seen.add(q_text)
            valid[level].append(q)

    return valid

def run_pipeline(
    path,
    questions_per_idea=2,       # Number of questions to generate per extracted idea
    ideas_per_chunk=2,          # Number of main ideas to extract per text chunk
    max_api_calls=30,           # Hard limit on Gemini API calls (quota control)
    output_file="questions.json" # File to save final grouped questions
):
    """
    Full end-to-end pipeline for generating multiple-choice questions from a document.

    Steps:
    1. Load and chunk the input text.
    2. Build a FAISS index for semantic passage retrieval.
    3. Extract main ideas from each chunk (batched).
    4. Retrieve top passages for each idea (batched).
    5. Generate multiple-choice questions grouped by difficulty (batched).
    6. Filter out low-quality or duplicate questions.
    7. Save results as JSON.
    """

    # Step 0: Load raw text from PDF/TXT
    text = load_document(path)

    # Step 1: Split text into ~50 adaptive chunks for processing
    chunks = chunk_text(text, target_chunks=50)

    # Step 2: Build FAISS vector index for semantic search
    index = build_index(chunks)

    # Keep track of API quota/budget
    budget = {"calls": max_api_calls}

    # Step 3: Extract main ideas (batched to reduce API calls)
    main_ideas = extract_main_ideas_batch(
        chunks,
        ideas_per_chunk,
        batch_size=5,   # Process 5 chunks per batch
        budget=budget
    )

    # Step 4: Retrieve top-k relevant passages for each idea
    idea_passage_pairs = retrieve_passages_batch(
        main_ideas,
        index,
        k=3,            # Get 3 supporting passages per idea
        batch_size=5
    )

    # Step 5: Generate multiple-choice questions grouped by difficulty
    mcqs = generate_mcqs_batch(
        idea_passage_pairs,
        questions_per_idea,
        batch_size=2,   # Generate 2 questions per batch
        budget=budget
    )

    # Step 6: Post-process to remove duplicates & weak questions
    mcqs = filter_grouped_questions(mcqs)

    # Step 7: Save final results to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mcqs, f, indent=2)

    # Print summary
    print(f"Pipeline complete. {sum(len(v) for v in mcqs.values())} high-quality questions saved to {output_file}")
    return mcqs


# ======================
# Example run
# ======================
FILE_PATH = "/Documents/transcript_1.txt"

# Run pipeline on a sample file
results = run_pipeline(
    FILE_PATH,
    questions_per_idea=2,   # Generate 2 questions per idea
    max_api_calls=20,       # Cap API calls for budget control
    output_file="questions.json"
)