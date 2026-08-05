from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader(
    "data",
    glob="**/*.pdf",
    loader_cls=UnstructuredPDFLoader,
    show_progress=True,
    use_multithreading=True
)

docs = loader.load()

print(f"Loaded {len(docs)} documents from the 'data' directory.")

MARKDOWN_SEPARATOR = "\n\n---\n\n"   ##doan nay con chua toi uu voi cac file pdf

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separator=MARKDOWN_SEPARATOR
)

splits = text_splitter.split_documents(docs)
print(splits)

