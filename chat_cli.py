from app.config import settings
from app.logger import configure_logging
from app.services.answer_services import generate_answer
from app.services.retrieval_services import retrieve


def main() -> None:
    configure_logging()
    

    history: list[dict[str, str]] = []

    print("Production RAG CLI")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        try:
            _, documents = retrieve(question, history)
            answer, sources = generate_answer(
                question,
                history,
                documents,
            )

            print(f"\nAssistant: {answer}")

            if sources:
                print("\nSources:")
                for source in sources:
                    location = source["source"]
                    if source["page"] is not None:
                        location += f", page {source['page']}"

                    print(
                        f"- [{source['label']}] {location} "
                        f"(chunk: {source['chunk_id']})"
                    )

            print()

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

            # Bound the in-memory CLI history.
            history = history[-settings.max_history_turns * 2:]

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()