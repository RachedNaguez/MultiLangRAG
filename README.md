# MultiLangRAG

Multilingual Retrieval-Augmented Generation (RAG) system. Ask questions in any language and get answers from a multilingual knowledge base powered by cross-lingual embeddings.

<p align="center">
  <img src="docs/MultiLangRAG.png" alt="MultiLangRAG Architecture" width="700">
</p>

## Features

- **Multilingual embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` maps 50+ languages into a shared vector space
- **Persistent vector store** — ChromaDB embeddings survive restarts (no re-embedding on every run)
- **Fast startup** — embedding model loaded once and cached; persisted embeddings skip re-indexing
- **Groq LLM** — uses Llama 3.3 70B for fast, high-quality generation
- **Rich CLI** — styled banner, colored prompts, markdown-rendered answers, source document display, and session status

## Architecture

```
User Query (any language)
        │
        ▼
┌───────────────────┐
│  Multilingual     │  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
│  Embeddings       │  Maps query + docs to shared vector space
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ChromaDB         │  Persistent local vector store
│  Retriever (k=2)  │  Fetches top-2 most relevant documents
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Groq LLM         │  Llama 3.3 70B Versatile
│  (answer gen)     │  Generates answer in the user's language
└───────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [Groq API key](https://console.groq.com/keys)

### Install

```bash
# Clone the repo
git clone <your-repo-url>
cd MultiLangRAG

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Configure

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

### Run

```bash
# Interactive mode (with Rich UI)
python main.py

# Single query
python main.py "How do I handle urgent IT issues?"

# French query
python main.py "Combien de jours de conges payes puis-je prendre ?"

# Spanish query
python main.py "¿Cuántos días de vacaciones tengo al año?"
```

### Interactive Commands

| Command                 | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `/history`            | Show all past queries and answers in a table   |
| `/sources`            | Show documents retrieved for the last query    |
| `/clear`              | Clear the screen                               |
| `/help`               | Display available commands                     |
| `/status`             | Show session info (model, uptime, query count) |
| `/quit` or `Ctrl+C` | Exit the application                           |

### CLI Options

```
python main.py [OPTIONS] [QUESTION]

Options:
  --model TEXT          LLM model name (default: llama-3.3-70b-versatile)
  --top-k INT          Documents to retrieve (default: 2)
  --temperature FLOAT  LLM temperature (default: 0.0)
  --persist-dir TEXT   ChromaDB persistence directory (default: data/chroma)
  -h, --help           Show this help message
```

### Example Session

```
╭──────────────────────────────────────────────────────────────╮
│  MultiLangRAG  Multilingual Retrieval-Augmented Generation   │
╰──────────────────────────────────────────────────────────────╯
  > Ask questions in any language
  > /history  past queries
  > /sources  last retrieved documents
  > /clear     clear screen
  > /quit      exit

  Session started 14:32:05 | Model: llama-3.3-70b-versatile

  [success]Ready![/success]

  > How do I handle urgent IT issues?

╭── Answer  (2 sources: Spanish) ──╮
│ For urgent IT issues, call the   │
│ emergency number instead of      │
│ sending an email.                │
╰──────────────────────────────────╯
  (2.3s) #1

  > /sources

  ┌───┬──────────┬────────────┬──────────────────────┐
  │ # │ Language │ Source     │ Content              │
  ├───┼──────────┼────────────┼──────────────────────┤
  │ 1 │ Spanish  │ IT_Policy  │ El soporte tecnico   │
  │ 2 │ English  │ HR_Policy  │ The company policy   │
  └───┴──────────┴────────────┴──────────────────────┘
```

## Project Structure

```
MultiLangRAG/
├── main.py                  # Entry point
├── pyproject.toml           # Project metadata and dependencies
├── .env                     # API keys (git-ignored)
├── .gitignore
├── .python-version
├── src/
│   └── multilangrag/
│       ├── __init__.py      # Package init
│       ├── config.py        # Dataclass configuration
│       ├── documents.py     # Multilingual document store
│       ├── embeddings.py    # Cached embedding model
│       ├── rag.py           # RAG chain builder
│       ├── vectorstore.py   # ChromaDB with persistence
│       └── cli.py           # Rich CLI with interactive mode
├── .env.example              # Template for API keys
├── data/
│   └── chroma/              # Persisted ChromaDB (git-ignored)
└── tests/
    └── test_rag.py          # Unit tests
```

## How It Works

1. **Embedding** — Documents and queries are embedded using a multilingual sentence transformer that maps 50+ languages into the same vector space. A French query can match an English document if they're semantically similar.
2. **Retrieval** — ChromaDB stores document vectors locally. On query, it returns the top-k most similar documents via cosine similarity.
3. **Generation** — The retrieved context and user question are passed to Llama 3.3 70B via Groq. The prompt instructs the LLM to answer in the same language as the question.

## Optimization Notes

| Technique                      | What it does                                                           |
| ------------------------------ | ---------------------------------------------------------------------- |
| **Singleton embeddings** | HuggingFace model loaded once, reused across calls (`embeddings.py`) |
| **Chroma persistence**   | Embeddings saved to`data/chroma/` — skip re-embedding on restart    |
| **Thread-safe init**     | Double-checked locking for embedding singleton                         |
| **Frozen config**        | `@dataclass(frozen=True)` prevents accidental mutation               |
| **Lazy store init**      | Vector store created on first access, not at import time               |

## Dependencies

| Package                   | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| `langchain`             | Core RAG framework                              |
| `langchain-chroma`      | ChromaDB vector store integration               |
| `langchain-groq`        | Groq LLM integration                            |
| `langchain-huggingface` | HuggingFace embeddings                          |
| `rich`                  | Terminal UI — colors, panels, tables, spinners |
| `python-dotenv`         | Environment variable loading                    |
| `sentence-transformers` | Multilingual embedding models                   |

## Testing

```bash
python -m pytest tests/ -v
```

## License

MIT
