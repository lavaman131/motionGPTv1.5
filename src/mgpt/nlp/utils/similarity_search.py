from torch import embedding
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from mgpt.nlp import PRECOMPUTED_DIR
from pathlib import Path
from mgpt.nlp import FileName

CLIP_MAX_TOKENS = 77  # max CLIP token length


def init_vector_db(
    text_file: str,
    precomputed_dir: FileName,
    chunk_size: int = CLIP_MAX_TOKENS,
    chunk_overlap: int = 0,
) -> Chroma:
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    precomputed_dir = Path(precomputed_dir)
    persist_directory = precomputed_dir.joinpath("chroma_db")

    if persist_directory.is_dir():
        # load from disk
        db = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )
    else:
        # create a new one
        loader = TextLoader(precomputed_dir.joinpath(text_file))
        documents = loader.load()

        # split it into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n"
        )
        docs = text_splitter.split_documents(documents)

        db = Chroma.from_documents(
            docs,
            persist_directory=str(persist_directory),
            embedding=embedding_function,
        )
    return db


if __name__ == "__main__":
    db = init_vector_db("action_vocab.txt", PRECOMPUTED_DIR)
    query = "Can you generate a jumping jack?"
    # get chunks
    docs = db.similarity_search(query, k=4)
    print(docs[0].page_content)
