import os
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class HybridRAGPipeline:
    """RAG with integrated hybrid search (semantic + keyword)"""

    def __init__(self, db_path="db", semantic_weight=0.5, keyword_weight=0.5):
        print("Initializing Hybrid RAG Pipeline...")
    
        # Initialize embedding model
        self.model = SentenceTransformer(
            'BAAI/bge-small-en-v1.5'
        )
        print("Embedding model loaded")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path
        )
        self.collection = self.client.get_collection(
            name="ap_elections"
        )

        print("ChromaDB loaded")

        # Initialize OpenAI
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print("OpenAI client initialized")

        # Hybrid search weights
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        print(
            f"Hybrid search weights: "
            f"Semantic {semantic_weight*100}% + "
            f"Keyword {keyword_weight*100}%"
        )

        # Initialize BM25
        self._initialize_bm25()

        print("Hybrid RAG Pipeline Ready!\n")

    def _initialize_bm25(self):
        """Initialize BM25 keyword search"""
        try:
            results = self.collection.get(
                include=["documents", "metadatas"]
            )

            if not results or not results['documents']:

                print("No documents found for BM25")
                self.bm25 = None
                self.documents_list = []
                self.metadata_list = []
                return
            self.documents_list = results['documents']
            self.metadata_list = results['metadatas']

            # Tokenize documents
            tokenized = [
                doc.lower().split()
                for doc in self.documents_list
            ]

            # Initialize BM25
            self.bm25 = BM25Okapi(tokenized)

            print(
                f"BM25 initialized with "
                f"{len(self.documents_list)} documents"
            )

        except Exception as e:

            print(f"Error initializing BM25: {e}")

            self.bm25 = None
            self.documents_list = []
            self.metadata_list = []

    def semantic_search(self, query, k=3):
        """Semantic search using embeddings"""

        try:
            query_embedding = self.model.encode(query)

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            similarities = []
            if results['distances'][0]:

                for distance in results['distances'][0]:
                    similarity = 1 / (1 + distance)
                    similarities.append(similarity)

            return {
                'documents': (
                    results['documents'][0]
                    if results['documents']
                    else []
                ),

                'metadatas': (
                    results['metadatas'][0]
                    if results['metadatas']
                    else []
                ),

                'scores': similarities
            }
        except Exception as e:

            print(f"Error in semantic search: {e}")
            return {
                'documents': [],
                'metadatas': [],
                'scores': []
            }

    def keyword_search(self, query, k=3):
        """Keyword search using BM25"""
        try:
            if self.bm25 is None:
                return {
                    'documents': [],
                    'metadatas': [],
                    'scores': []
                }
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(
                tokenized_query
            )
            top_k_indices = np.argsort(scores)[::-1][:k]
            documents = [
                self.documents_list[i]
                for i in top_k_indices
                if i < len(self.documents_list)
            ]

            metadatas = [
                self.metadata_list[i]
                for i in top_k_indices
                if i < len(self.metadata_list)
            ]

            bm25_scores = [
                scores[i]
                for i in top_k_indices
            ]

            # Normalize scores
            if max(bm25_scores) > 0:

                normalized_scores = [
                    s / max(bm25_scores)
                    for s in bm25_scores
                ]

            else:
                normalized_scores = bm25_scores
            return {
                'documents': documents,
                'metadatas': metadatas,
                'scores': normalized_scores
            }
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return {
                'documents': [],
                'metadatas': [],
                'scores': []
            }
        
    def rewrite_query(self, query):
        """Use LLM to resolve ambiguous or vague intent before retrieval"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=120,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query rewriter for an Andhra Pradesh elections RAG system "
                            "covering elections held in 2004, 2008, 2014, 2019, and 2024.\n\n"
                            "Your job is to rewrite the user's question into a precise, unambiguous "
                            "retrieval query that can be matched against election documents.\n\n"
                            "Guidelines:\n"
                            "- Understand the intent of the question fully before rewriting.\n"
                            "- If the question refers to time vaguely (e.g. recently, latest, current, "
                            "last, nowadays, most recent, now), resolve it to the most recent election "
                            "year covered: 2024.\n"
                            "- If the question already names a specific year or election, keep it.\n"
                            "- If the question is a comparison or trend query, keep all years intact.\n"
                            "- Preserve the original question's intent — do not change what is being asked.\n"
                            "- Return ONLY the rewritten query. No explanation, no quotes, no extra text."
                        )
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten and rewritten != query:
                print(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"Query rewrite failed, using original: {e}")
            return query

    def hybrid_search(self, query, k=10):
        """Hybrid retrieval combining semantic + keyword search"""

        # Semantic search
        semantic_results = self.semantic_search(
            query,
            k=k
        )
        # Keyword search
        keyword_results = self.keyword_search(
            query,
            k=k
        )
        combined = {}
        # Semantic results
        for i, doc in enumerate(
            semantic_results['documents']
        ):
            score = (
                semantic_results['scores'][i]
                * self.semantic_weight
            )
            if doc not in combined:
                combined[doc] = {
                    'score': 0,
                    'metadata': (
                        semantic_results['metadatas'][i]
                        if i < len(
                            semantic_results['metadatas']
                        )
                        else {}
                    )
                }
            combined[doc]['score'] += score
        # Keyword results
        for i, doc in enumerate(
            keyword_results['documents']
        ):
            score = (
                keyword_results['scores'][i]
                * self.keyword_weight
            )
            if doc not in combined:
                combined[doc] = {
                    'score': 0,
                    'metadata': (
                        keyword_results['metadatas'][i]
                        if i < len(
                            keyword_results['metadatas']
                        )
                        else {}
                    )
                }
            combined[doc]['score'] += score
        # Threshold filtering
        threshold = 0.25
        filtered = {
            doc: data
            for doc, data in combined.items()
            if data['score'] >= threshold
        }
        # Sort results
        sorted_results = sorted(
            filtered.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:k]
        documents = [
            doc for doc, data in sorted_results
        ]
        metadatas = [
            data['metadata']
            for doc, data in sorted_results
        ]
        scores = [
            data['score']
            for doc, data in sorted_results
        ]
        return {
            'documents': documents,
            'metadatas': metadatas,
            'scores': scores
        }

    def get_context(self, query, k=3):
        """Get retrieved context"""
        results = self.hybrid_search(
            query,
            k
        )
        if not results['documents']:
            return None, None
        context = "\n\n---\n\n".join(
            results['documents']
        )
        sources = [
            {**meta, "chunk_text": doc}
            for meta, doc in zip(
                results['metadatas'],
                results['documents']
            )
        ]
        return context, sources

    def generate_answer(self, query):
        """Generate grounded response"""
        try:
            print(f"\nQuery: {query}")
            query = self.rewrite_query(query)
            print("Searching with hybrid retrieval...")
            context, sources = self.get_context(
                query,
                k=5
            )
            if context is None:
                print("No relevant context found")
                return (
                    "Sorry, I couldn't find relevant information.",
                    []
                )
            print("Context retrieved")

            # System prompt
            system_prompt = """
You are an expert Retrieval-Augmented Generation (RAG) assistant specialized in Andhra Pradesh elections from 2004–2024.

Your task is to answer questions ONLY using the retrieved context provided.

STRICT RULES:
1. Do NOT invent, assume, or hallucinate information.
2. If the answer is not present in the retrieved context, clearly say:
   "I could not find sufficient information in the retrieved election data."
3. Do NOT use external knowledge.
4. Prefer factual, grounded, and concise responses.
5. Always prioritize retrieved evidence over assumptions.

ANSWERING GUIDELINES:
- Mention election years whenever relevant.
- Mention political parties clearly.
- Include vote share, seat counts, turnout, or statistics if available.
- Support claims using retrieved evidence.
- For comparison questions, compare all requested years separately.
- For trend questions, summarize patterns across elections.
- If multiple sources disagree, mention the conflict instead of guessing.

RESPONSE STYLE:
- Be structured and readable.
- Use bullet points when appropriate.
- Keep answers informative but concise.
- Focus on factual political analysis from the retrieved context only.

Your goal is to provide accurate, grounded, retrieval-based answers with high factual reliability.
"""

            # User prompt
            user_message = f"""
You are given retrieved Andhra Pradesh election data.

Use ONLY the provided context to answer the question accurately.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer only from the retrieved context.
- Do not invent facts or use outside knowledge.
- If information is missing, clearly mention it.
- For comparison questions, compare each election year separately.
- Include relevant statistics such as:
  • seats won
  • vote share
  • voter turnout
  • alliances
  • party performance
- Mention political parties and election years clearly.
- Keep the answer structured and concise.
- Use bullet points where helpful.

Provide:
1. Direct answer
2. Supporting evidence from context
3. Relevant election years
4. Political parties involved
5. Important statistics if available
"""

            print("Generating answer...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=500,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )
            answer = response.choices[0].message.content
            print("Answer generated successfully")
            return answer, sources
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: {str(e)}", []

    def display_sources(self, sources):
        """Display retrieved sources"""
        if not sources:
            return
        print("\nSOURCES")
        for i, source in enumerate(sources, 1):
            year = source.get('year', 'N/A')
            source_type = source.get('type', 'N/A')
            chunk = source.get('chunk_number', 'N/A')
            print(
                f"{i}. "
                f"Year: {year} | "
                f"Type: {source_type} | "
                f"Chunk: {chunk}"
            )


def main():
    print("HYBRID AP ELECTIONS RAG")
    rag = HybridRAGPipeline()

    while True:
        try:
            query = input(
                "\nAsk a question (or type 'exit'): "
            ).strip()
            if query.lower() == "exit":
                print("\nGoodbye!")
                break
            if not query:
                print("Please enter a question")
                continue

            answer, sources = rag.generate_answer(
                query
            )
            print("ANSWER")
            print(answer)
            rag.display_sources(sources)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()