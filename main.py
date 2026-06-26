"""MultiLangRAG — entry point.

Usage:
    python main.py                    # interactive mode
    python main.py "your question"    # single query
    python main.py --help             # all options
"""

from multilangrag.cli import main

if __name__ == "__main__":
    main()
