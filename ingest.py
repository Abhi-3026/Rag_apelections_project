import os
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb


class DataIngestion:
    def __init__(self, data_dir="data", db_path="db"):

        self.data_dir = data_dir
        self.db_path = db_path

        # Initialize embedding model
        self.model = SentenceTransformer(
            'BAAI/bge-small-en-v1.5'
        )

        # Initialize Recursive Text Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )

        # Store chunks for BM25 / hybrid retrieval
        self.documents_list = []

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path
        )

        # Delete old collection if exists
        try:
            self.client.delete_collection(
                name="ap_elections"
            )
        except:
            pass

        # Create new collection
        self.collection = self.client.create_collection(
            name="ap_elections",
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Initialized ChromaDB at: {db_path}")

    def load_csv_files(self):
        """Load all CSV election data"""
        print("STEP 1: LOADING CSV FILES")

        csv_data = []

        csv_files = list(Path(self.data_dir).glob("*/*.csv"))

        print(f"\nFound {len(csv_files)} CSV files")

        for csv_file in sorted(csv_files):
            print(f"\n📖 Loading: {csv_file}")

            try:
                df = pd.read_csv(csv_file)

                print(f"   - Rows: {len(df)}")
                print(f"   - Columns: {len(df.columns)}")

                text = self._convert_csv_to_text(df, csv_file.parent.name)
                csv_data.append({
                    'year': csv_file.parent.name,
                    'source': str(csv_file),
                    'content': text,
                    'type': 'election_results'
                })

                print("Processed successfully")
            except Exception as e:
                print(f"Error: {e}")

        print(f"\nLoaded {len(csv_data)} CSV files")
        return csv_data

    def load_text_files(self):
        """Load all analysis text files"""
        print("STEP 2: LOADING TEXT ANALYSIS FILES")

        text_data = []
        txt_files = list(Path(self.data_dir).glob("*/*.txt"))
        print(f"\nFound {len(txt_files)} text files")

        for txt_file in sorted(txt_files):
            print(f"\nLoading: {txt_file}")

            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                print(f"Size: {len(content)} characters")

                text_data.append({
                    'year': txt_file.parent.name,
                    'source': str(txt_file),
                    'content': content,
                    'type': 'analysis'
                })

                print("Loaded successfully")
            except Exception as e:
                print(f"Error: {e}")

        print(f"\nLoaded {len(text_data)} text files")
        return text_data

    def _convert_csv_to_text(self, df, year):
        """Convert CSV dataframe into semantic readable text"""

        text = f"ANDHRA PRADESH ELECTION RESULTS - {year}\n"

        text += f"Total Records: {len(df)}\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"

        text += "DETAILED ELECTION DATA\n"

        for idx, row in df.iterrows():
            text += f"Record {idx + 1}:\n"
            text += "-"*40 + "\n"

            for col, value in row.items():
                text += f"{col}: {value}\n"
            text += "\n"
        return text

    def chunk_document(self, text, chunk_size=700, overlap=100):
        """Manual chunking method"""
        chunks = []

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def create_embeddings(self, all_data):
        """Create embeddings for all chunks"""

        print("STEP 3: CREATING CHUNKS + EMBEDDINGS")

        documents = []
        metadata_list = []
        ids = []
        embeddings = []
        total_chunks = 0

        print(f"\nProcessing {len(all_data)} documents...")

        for idx, doc in enumerate(all_data):
            print(f"\n[{idx+1}/{len(all_data)}] Processing: {doc['year']} - {doc['type']}")

            try:
                # Recursive chunking
                chunks = self.splitter.split_text(
                    doc['content']
                )

                print(f"Created {len(chunks)} chunks")
                for chunk_idx, chunk in enumerate(chunks):
                    # Create embedding
                    embedding = self.model.encode(chunk)
                    # Create unique chunk ID
                    doc_id = (
                        f"{doc['year']}_{doc['type']}_{idx}_{chunk_idx}"
                    )

                    documents.append(chunk)
                    embeddings.append(
                        embedding.tolist()
                    )

                    ids.append(doc_id)

                    metadata_list.append({
                        'year': doc['year'],
                        'type': doc['type'],
                        'source': doc['source'],
                        'chunk_number': chunk_idx
                    })

                    # Store chunk for BM25
                    self.documents_list.append(chunk)

                    total_chunks += 1

                print("Embeddings created successfully")

            except Exception as e:

                print(f"Error: {e}")

        print(f"\nTotal Chunks Created: {total_chunks}")

        print(f"Total Embeddings Created: {len(embeddings)}")

        return documents, embeddings, ids, metadata_list

    def store_in_chromadb(self, documents, embeddings, ids, metadata_list):
        """Store embeddings in ChromaDB"""
        print("STEP 4: STORING IN CHROMADB")

        print(f"\nAdding {len(documents)} chunks to vector database...")

        try:
            batch_size = 20

            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))

                self.collection.add(
                    ids=ids[i:batch_end],
                    documents=documents[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    metadatas=metadata_list[i:batch_end]
                )

                print(f"Added {batch_end}/{len(documents)} chunks")

            print(f"\nSuccessfully stored all chunks in ChromaDB")

        except Exception as e:
            print(f"Error storing in ChromaDB: {e}")
            raise

    def ingest_all(self):
        """Run complete ingestion pipeline"""

        print("AP ELECTIONS RAG - DATA INGESTION PIPELINE")
        # STEP 1
        csv_data = self.load_csv_files()
        # STEP 2
        text_data = self.load_text_files()
        # Combine all documents
        all_data = csv_data + text_data
        print(f"TOTAL DOCUMENTS LOADED: {len(all_data)}")
        # STEP 3
        documents, embeddings, ids, metadata = self.create_embeddings(all_data)

        # STEP 4
        self.store_in_chromadb(documents, embeddings, ids, metadata)
        print("INGESTION COMPLETE!")

        print("\n📊 SUMMARY:")
        print(f"  • CSV Files: {len(csv_data)}")
        print(f"  • Text Files: {len(text_data)}")
        print(f"  • Total Documents: {len(all_data)}")
        print(f"  • Total Chunks: {len(documents)}")
        print(f"  • Embeddings Created: {len(embeddings)}")
        print(f"  • Database Location: {self.db_path}/")
        print(f"  • Collection Name: ap_elections")

        print("\n🚀 NEXT STEP:")
        print("   Run: streamlit run app.py")

if __name__ == "__main__":
    ingestor = DataIngestion(data_dir="data", db_path="db")
    ingestor.ingest_all()