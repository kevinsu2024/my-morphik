#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz 
import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from PIL import Image
from psycopg2 import extras
from tqdm import tqdm

from base_eval import BaseRAGEvaluator
from colpali_engine.models import ColPali, ColPaliProcessor
from sentence_transformers import SentenceTransformer

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.models.chunk import DocumentChunk
from core.reranker.flag_reranker import FlagReranker

# Remove the project root from Python path after imports
sys.path.pop(0)

# Load environment variables
load_dotenv('../../.env.example', override=True)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def completion(**kwargs):
    """Wrapper for OpenAI completion API with proper parameter names"""
    kwargs.pop('api_key', None)
    
    if 'max_tokens' in kwargs:
        kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
    return client.chat.completions.create(**kwargs)

# Global debug flag
DEBUG_MODE = False

class ColPaliDocumentPage:
    """Document page with multi-vector embeddings from ColPali."""
    
    def __init__(self, document_id: str, page_number: int, image_path: str, 
                 content: str = '', page_id: int = None, embeddings: np.ndarray = None):
        self.document_id = document_id
        self.page_number = page_number
        self.image_path = image_path
        self.content = content
        self.page_id = page_id
        self.embeddings = embeddings
        self.chunk_number = f"{document_id}_page_{page_number}"

class KevinEvaluator(BaseRAGEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Device selection with proper dtype handling
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_mps = True
            self.model_dtype = torch.float16
            print("✓ MPS (Apple Silicon GPU) available - using float16 for compatibility")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.use_mps = False
            self.model_dtype = torch.bfloat16
            print("✓ CUDA available - using bfloat16")
        else:
            self.device = torch.device("cpu")
            self.use_mps = False
            self.model_dtype = torch.float32
            print("Using CPU - using float32")
        
        self.model = None
        self.processor = None
        self.embedding_model = None
        self.images_dir = None
        self.embedding_cache_dir = Path("./.embedding_cache")
        self.embedding_cache_dir.mkdir(exist_ok=True)
        self.debug_dir = Path("./debug_output")
        self.debug_dir.mkdir(exist_ok=True)
        
    def setup_client(self, **kwargs) -> dict:
        """Initialize the RAG system client with memory optimizations."""
        
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI environment variable not set.")
        
        try:
            conn = psycopg2.connect(postgres_uri)
            register_vector(conn)
            print("✓ Connected to PostgreSQL successfully")
            
            self._setup_database(conn)

        except Exception as e:
            raise ConnectionError(f"Error connecting to PostgreSQL: {e}")

        try:
            print("Loading embedding and reranking models...")
            
            # Load the text embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print(f"    ✓ SentenceTransformer text embedding model loaded successfully on {self.device}")

            # Load the multi-modal model
            model_name = "vidore/colpali-v1.3"
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=True
            ).to(self.device).eval()
            print(f"    ✓ ColPali multi-modal model loaded successfully on {self.device}")

        except Exception as e:
            print(f"⌐ Complete model loading failure: {e}")
            raise

        try:
            reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
            print("✓ Re-ranker initialized successfully")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize reranker: {e}")
            reranker = None
            loop = None

        self.images_dir = Path("./document_images")
        self.images_dir.mkdir(exist_ok=True)

        return {
            "db_conn": conn, 
            "reranker": reranker, 
            "loop": loop,
            "model": self.model,
            "processor": self.processor
        }
    
    def _setup_database(self, conn: psycopg2.extensions.connection):
        """Setup database tables for the hybrid knowledge graph."""
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS hstore;") # Often useful with JSONB

            # Drop existing tables to ensure a clean schema
            cur.execute("DROP TABLE IF EXISTS graph_edges CASCADE;")
            cur.execute("DROP TABLE IF EXISTS graph_nodes CASCADE;")
            cur.execute("DROP TABLE IF EXISTS document_pages CASCADE;")

            # Table for high-level document metadata
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_pages (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL UNIQUE,
                    image_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            )

            # Core table for graph nodes, now with a flexible JSONB column
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    page_number INTEGER,
                    node_type VARCHAR(50), -- e.g., 'text_chunk', 'image_chunk'
                    content TEXT,          -- The raw text chunk or a description of the image
                    properties JSONB,      -- For structured data like embeddings, metrics, etc.
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            )

            # Table for relationships between nodes
            cur.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id SERIAL PRIMARY KEY,
                    source_node_id INTEGER REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
                    target_node_id INTEGER REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
                    relationship_label VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            )

            # Indexing for performance
            cur.execute("CREATE INDEX IF NOT EXISTS nodes_doc_id_idx ON graph_nodes (doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS nodes_properties_idx ON graph_nodes USING gin(properties);") # GIN index for JSONB
            cur.execute("CREATE INDEX IF NOT EXISTS edges_source_idx ON graph_edges (source_node_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS edges_target_idx ON graph_edges (target_node_id);")
            
            conn.commit()

    def ingest(self, client: dict, docs_dir: Path, **kwargs) -> List[str]:
        """Ingests documents by extracting a knowledge graph and storing it."""
        print(f"Ingesting documents from: {docs_dir}")
        doc_files = list(docs_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No PDF files found in {docs_dir}")

        conn = client["db_conn"]
        for doc_file in tqdm(doc_files, desc="Ingesting Documents", unit="doc"):
            print(f"\n  - Processing {doc_file.name} into knowledge graph...")
            full_text = "\n".join(self._extract_enhanced_text_data(doc_file)[2])
            
            graph = self._extract_graph_from_document(full_text)
            if graph and graph.get("nodes") and graph.get("edges"):
                self._insert_graph_into_db(conn, doc_file, graph)
            else:
                print(f"    - Could not extract a valid graph from {doc_file.name}")

        print(f"\n✓ Successfully ingested {len(doc_files)} documents into the graph.")
        return [doc.name for doc in doc_files]

    def _extract_graph_from_document(self, document_text: str) -> Optional[Dict[str, List]]:
        """Uses a powerful LLM prompt to extract a full knowledge graph from a document."""
        prompt = f"""You are a high-precision financial analyst. Your task is to extract a detailed knowledge graph from the provided text from a financial document. 

Extract the following node types: 
- Organization (e.g., \"Palantir\", \"NVIDIA\")
- Metric (e.g., \"Total Revenue\", \"US Commercial Customer Count\")
- TimePeriod (e.g., \"Q1 2025\", \"Fiscal Year 2026\")
- Value (e.g., \"$634M\", \"83%\")
- TextChunk (A meaningful chunk of text that contains one or more entities)

Extract the following relationship types:
- HAS_METRIC (Organization -> Metric)
- HAS_VALUE (Metric -> Value)
- IN_PERIOD (Value -> TimePeriod)
- DESCRIBED_IN (Any entity -> TextChunk)

First, create all the TextChunk nodes for the document. Then, create the specific entities (Organization, Metric, TimePeriod, Value) and link them to the TextChunk they came from using the DESCRIBED_IN relationship.

Respond ONLY with a single JSON object with two keys: "nodes" and "edges". The nodes should have a unique "id" (a simple integer starting from 1), a "type", and "properties" (e.g., {{'name': 'Total Revenue'}} or {{'amount': 634, 'unit': 'million'}}).

EXAMPLE:
Input Text: \"In Q1 2025, Palantir's total revenue grew to $634M.\"
Output JSON:
{{
  "nodes": [
    {{ "id": 1, "type": "TextChunk", "properties": {{'content': 'In Q1 2025, Palantir's total revenue grew to $634M.'}} }},
    {{ "id": 2, "type": "Organization", "properties": {{'name': 'Palantir'}} }},
    {{ "id": 3, "type": "Metric", "properties": {{'name': 'Total Revenue'}} }},
    {{ "id": 4, "type": "TimePeriod", "properties": {{'value': 'Q1 2025'}} }},
    {{ "id": 5, "type": "Value", "properties": {{'amount': 634, 'unit': 'million'}} }}
  ],
  "edges": [
    {{ "source": 2, "target": 3, "label": "HAS_METRIC" }},
    {{ "source": 3, "target": 5, "label": "HAS_VALUE" }},
    {{ "source": 5, "target": 4, "label": "IN_PERIOD" }},
    {{ "source": 2, "target": 1, "label": "DESCRIBED_IN" }},
    {{ "source": 3, "target": 1, "label": "DESCRIBED_IN" }},
    {{ "source": 4, "target": 1, "label": "DESCRIBED_IN" }},
    {{ "source": 5, "target": 1, "label": "DESCRIBED_IN" }}
  ]
}}

DOCUMENT TEXT:
{document_text[:15000]}  # Limit context to avoid exceeding model limits

JSON OUTPUT:
"""
        try:
            print("    - Extracting graph with LLM...")
            response = completion(
                model="o4-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            graph_data = response.choices[0].message.content
            if graph_data:
                return json.loads(graph_data)
            return None
        except Exception as e:
            print(f"    ⚠️ LLM call for graph extraction failed: {e}")
            return None

    def _insert_graph_into_db(self, conn: psycopg2.extensions.connection, doc_file: Path, graph: Dict[str, List]):
        """Inserts nodes and edges from an extracted graph into the database."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        temp_to_real_id_map = {}

        with conn.cursor() as cur:
            # Insert nodes and get their real, database-generated IDs
            for node in nodes:
                temp_id = node["id"]
                node_type = node["type"]
                properties = node.get("properties", {})
                content = properties.get("content", properties.get("name", ""))
                
                cur.execute(
                    "INSERT INTO graph_nodes (doc_id, page_number, node_type, content, properties) VALUES (%s, %s, %s, %s, %s) RETURNING node_id",
                    (doc_file.name, 0, node_type, content, json.dumps(properties))
                )
                real_id = cur.fetchone()[0]
                temp_to_real_id_map[temp_id] = real_id
            print(f"    ✓ Stored {len(nodes)} nodes.")

            # Prepare edges with the real database IDs
            edges_to_insert = []
            for edge in edges:
                source_temp_id = edge["source"]
                target_temp_id = edge["target"]
                label = edge["label"]
                
                if source_temp_id in temp_to_real_id_map and target_temp_id in temp_to_real_id_map:
                    source_real_id = temp_to_real_id_map[source_temp_id]
                    target_real_id = temp_to_real_id_map[target_temp_id]
                    edges_to_insert.append((source_real_id, target_real_id, label))
            
            # Batch insert all edges
            if edges_to_insert:
                extras.execute_batch(cur, 
                    "INSERT INTO graph_edges (source_node_id, target_node_id, relationship_label) VALUES (%s, %s, %s)",
                    edges_to_insert
                )
                print(f"    ✓ Stored {len(edges_to_insert)} relationships.")

        conn.commit()

    def _convert_pdf_to_images_pymupdf(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images using PyMuPDF."""
        doc = fitz.open(pdf_path)
        images = []
        for page_num in tqdm(range(len(doc)), desc=f"  Converting {pdf_path.name}", unit="page", leave=False):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if img.size[0] > 1200 or img.size[1] > 1200:
                img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            images.append(img)
        doc.close()
        return images

    def _extract_enhanced_text_data(self, pdf_path: Path) -> tuple[List[str], List[str], List[str]]:
        """Enhanced text extraction with tables and structured text blocks."""
        doc = fitz.open(pdf_path)
        texts, tables_data, text_blocks = [], [], []
        
        for page in doc:
            texts.append(page.get_text().strip())
            
            page_tables = []
            try:
                for table in page.find_tables():
                    table_content = table.extract()
                    if table_content:
                        page_tables.append(self._format_table_content(table_content))
            except Exception:
                pass # Ignore table extraction errors
            tables_data.append("\n".join(page_tables))

            try:
                blocks = [b[4] for b in page.get_text("blocks") if b[4].strip()]
                text_blocks.append("\n\n".join(blocks))
            except Exception:
                text_blocks.append("")

        doc.close()
        return texts, tables_data, text_blocks

    def _format_table_content(self, table_content: List[List]) -> str:
        """Format table content into readable text."""
        return "\n".join([" | ".join([str(c).strip() for c in r if c is not None]) for r in table_content if r])

    def _combine_text_sources(self, text: str, tables: str, blocks: str) -> str:
        """Combine different text sources into coherent content."""
        sources = []
        if text and text.strip():
            sources.append(f"Text Content:\n{text.strip()}")
        if tables and tables.strip():
            sources.append(f"Table Data:\n{tables.strip()}")
        # Always include structured blocks if they exist, as they preserve layout
        if blocks and blocks.strip():
            sources.append(f"Structured Layout:\n{blocks.strip()}")
        
        return "\n\n".join(sources) if sources else "Visual elements with no extractable text."

    def query(self, client: dict, question: str, **kwargs) -> str:
        """Query using the hybrid GraphRAG engine."""
        if DEBUG_MODE:
            print(f"\n[DEBUG] === GRAPH QUERY START: {question} ===")

        try:
            # 1. Hybrid Search: Find initial relevant nodes by text and visual meaning.
            retrieved_nodes = self._hybrid_search_nodes(client, question)
            if not retrieved_nodes:
                return "Could not find any relevant content for this question."

            # 2. Graph Expansion: Broaden context by including connected nodes.
            expanded_nodes = self._expand_with_graph(client, retrieved_nodes)

            # 3. Rerank and Build Context: Prioritize and format the final context.
            reranked_nodes = self._rerank_nodes(client, question, expanded_nodes)
            context = self._build_graph_context(reranked_nodes)

            if DEBUG_MODE:
                debug_context_file = self.debug_dir / f"context_debug_{hash(question) % 10000}.txt"
                debug_context_file.write_text(f"Question: {question}\n\nContext:\n{context}", encoding='utf-8')
                print(f"[DEBUG] Context saved to: {debug_context_file}")

            # 4. Generate Final Answer
            return self._generate_answer(context, question)

        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
            if 'conn' in client: client['db_conn'].rollback()
            return "[ERROR] An exception occurred during query processing."

    def _hybrid_search_nodes(self, client: dict, question: str) -> List[Dict[str, Any]]:
        """Performs hybrid vector search for both text and visual nodes."""
        print("  - Step 1: Performing hybrid search for initial nodes...")
        
        # Generate embeddings for the query
        text_query_embedding = self._embed_text_chunk(question)
        visual_query_embedding = self._embed_image_chunk(None, client["model"], client["processor"]) # ColPali can embed text queries as well

        results = []
        with client["db_conn"].cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Text search
            cur.execute(
                """SELECT *, 1 - ((properties->>'embedding_vector')::vector <=> %s) AS similarity 
                   FROM graph_nodes WHERE properties->>'embedding_type' = 'text_chunk' 
                   ORDER BY similarity DESC LIMIT 5""",
                (str(text_query_embedding),)
            )
            results.extend(cur.fetchall())
            print(f"    ✓ Found {len(results)} initial text nodes.")

            # Visual search
            cur.execute(
                """SELECT *, 1 - ((properties->>'embedding_vector')::vector <=> %s) AS similarity 
                   FROM graph_nodes WHERE properties->>'embedding_type' = 'image_chunk' 
                   ORDER BY similarity DESC LIMIT 5""",
                (str(visual_query_embedding),)
            )
            visual_results = cur.fetchall()
            results.extend(visual_results)
            print(f"    ✓ Found {len(visual_results)} initial image nodes.")

        # Remove duplicates by node_id
        unique_results = {r['node_id']: r for r in results}
        print(f"    ✓ Combined search yields {len(unique_results)} unique nodes.")
        return list(unique_results.values())

    def _expand_with_graph(self, client: dict, initial_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expands the node set by including directly connected nodes from the graph."""
        print("  - Step 2: Expanding node set with graph connections...")
        if not initial_nodes:
            return []

        node_ids = {node['node_id'] for node in initial_nodes}
        expanded_node_ids = set(node_ids)

        with client["db_conn"].cursor() as cur:
            cur.execute(
                "SELECT target_node_id FROM graph_edges WHERE source_node_id = ANY(%s)",
                (list(node_ids),)
            )
            for row in cur.fetchall():
                expanded_node_ids.add(row[0])

        print(f"    ✓ Expanded from {len(node_ids)} to {len(expanded_node_ids)} nodes.")
        
        # Fetch full data for all nodes in the expanded set
        with client["db_conn"].cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM graph_nodes WHERE node_id = ANY(%s)", (list(expanded_node_ids),))
            return cur.fetchall()

    def _rerank_nodes(self, client: dict, question: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranks the collected nodes against the original question."""
        print("  - Step 3: Reranking all collected nodes...")
        reranker, loop = client.get("reranker"), client.get("loop")
        if not (reranker and loop and nodes):
            return nodes

        # The reranker expects objects with a 'content' attribute.
        class TempNode:
            def __init__(self, node_dict):
                self.content = node_dict['content']
                self.original_dict = node_dict

        temp_nodes = [TempNode(n) for n in nodes]
        reranked_temp_nodes = loop.run_until_complete(reranker.rerank(question, temp_nodes))
        
        reranked_nodes = [n.original_dict for n in reranked_temp_nodes]
        print(f"    ✓ Reranked {len(reranked_nodes)} nodes.")
        return reranked_nodes[:15] # Return top 15 for final context

    def _build_graph_context(self, nodes: List[Dict[str, Any]]) -> str:
        """Builds the final context string from the top-ranked nodes."""
        contexts = []
        for node in nodes:
            context_parts = [
                f"--- START CONTEXT: {node['doc_id']}, Page {node['page_number']}, Node {node['node_id']} ---",
                node['content'].strip(),
                f"--- END CONTEXT ---"
            ]
            contexts.append("\n".join(context_parts))
        return "\n\n".join(contexts)

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate the final answer using the LLM."""
        prompt = f'''You are an expert document analyst. Your task is to answer the user's question based *only* on the provided context.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided context.
2. If the answer requires calculation, provide the final numerical answer first, then briefly show the calculation used to arrive at it. Be precise with numbers and calculations
3. Cite page numbers and node IDs, like `(Page X, Node Y)`, for every piece of data you use.
4. If the context does not contain the information to answer the question, state only: "The provided context does not contain enough information to answer the question."

Context from Document Pages:
{context}

Question: {question}

Final Answer:'''

        try:
            response = completion(
                model="o4-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            final_answer = response.choices[0].message.content
        except Exception as e:
            print(f"    ⚠️ OpenAI API call failed: {e}")
            return "[ERROR] Failed to generate an answer from the language model."

        if DEBUG_MODE:
            print(f"\n[DEBUG] Final Generated Answer:\n{final_answer}\n")
        return final_answer

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = KevinEvaluator.create_cli_parser("kevin")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True
        print("\n[--- DEBUG MODE ENABLED ---]\n")

    print(f"✓ PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        print("✓ MPS available, using Apple Silicon GPU")
    else:
        print("✓ Using CPU")

    evaluator = KevinEvaluator(
        system_name="kevin",
        docs_dir=args.docs_dir,
        questions_file=args.questions,
        output_file=args.output,
    )
    
    try:
        output_file = evaluator.run_evaluation(skip_ingestion=args.skip_ingestion)
        print(f"\n🎉 Multi-vector evaluation completed successfully!")
        print(f"📄 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n⌐ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if hasattr(evaluator, "_client") and evaluator._client:
            client = evaluator._client
            if "db_conn" in client and client["db_conn"]:
                client["db_conn"].close()
            if "loop" in client and client["loop"]:
                client["loop"].close()
    return 0

if __name__ == "__main__":
    exit(main())
