from pathlib import Path

import chromadb


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE_CANDIDATES = [
    ROOT_DIR / "data" / "question_answer.txt",
    ROOT_DIR / "data" / "questions_answer.txt",
    ROOT_DIR / "data" / "questions_output.txt",
]
CHROMA_PATH = ROOT_DIR / "chroma"
DEFAULT_COLLECTION = "nutrition_qna"


def resolve_data_file():
    for candidate in DATA_FILE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No Q&A document found in data/. Tried: "
        + ", ".join(str(path) for path in DATA_FILE_CANDIDATES)
    )


def parse_question_answer_file(file_path: Path):
    """Read a text file containing Q&A entries separated by blank lines."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in content.split("\n\n") if block.strip()]

    questions_and_answers = []
    for block in blocks:
        question = ""
        answer = ""

        for line in block.splitlines():
            cleaned = line.strip()
            if cleaned.lower().startswith("question:"):
                question = cleaned.split(":", 1)[1].strip()
            elif cleaned.lower().startswith("answer:"):
                answer = cleaned.split(":", 1)[1].strip()

        if question and answer:
            document_text = f"Question: {question}\nAnswer: {answer}"
            questions_and_answers.append(
                {
                    "question": question,
                    "answer": answer,
                    "document": document_text,
                }
            )

    return questions_and_answers


def setup_question_answer_rag(
    file_path: Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Create a ChromaDB collection from the Q&A text file."""
    if file_path is None:
        file_path = resolve_data_file()
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "description": "Q&A database built from the question_answer.txt file"
        },
    )

    entries = parse_question_answer_file(file_path)
    if not entries:
        raise ValueError(f"No valid Q&A entries found in {file_path}")

    documents = [entry["document"] for entry in entries]
    metadatas = [
        {
            "question": entry["question"],
            "answer": entry["answer"],
            "source_file": file_path.name,
            "topic": "question_answer",
        }
        for entry in entries
    ]
    ids = [f"qa_{index}" for index in range(len(entries))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Added {len(entries)} Q&A entries to ChromaDB collection '{collection_name}'")
    return collection


if __name__ == "__main__":
    collection = setup_question_answer_rag()

    print("\nSample query test:")
    results = collection.query(
        query_texts=["How do I request approval for professional development courses?"],
        n_results=2,
    )

    for index, doc in enumerate(results["documents"][0], start=1):
        print(f"\nResult {index}:\n{doc}\n")
