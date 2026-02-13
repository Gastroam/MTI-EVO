"""
MTI Ghost Indexer
=================
Scans the local codebase to extract "Code DNA" (Symbols) for the Semantic Graph.
Uses AST for zero-runtime extraction.
"""
import ast
import os
import pathlib
from typing import List, Dict

class GhostIndexer:
    def __init__(self, root_dir: str):
        self.root_dir = pathlib.Path(root_dir)
        self.symbols = []

    def scan(self) -> List[Dict]:
        """
        Scans all .py files in root_dir and returns a list of symbols.
        Returns: [{'name': 'MyClass', 'type': 'class', 'file': 'utils.py'}, ...]
        """
        self.symbols = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".py") and "venv" not in root and ".git" not in root:
                    full_path = os.path.join(root, file)
                    self._parse_file(full_path)
        return self.symbols

    def _parse_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            
            rel_path = os.path.relpath(path, self.root_dir)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.symbols.append({
                        "name": node.name,
                        "type": "class",
                        "file": rel_path,
                        "line": node.lineno
                    })
                    # Scan methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                            self.symbols.append({
                                "name": f"{node.name}.{item.name}",
                                "type": "method",
                                "file": rel_path,
                                "line": item.lineno
                            })
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                     # Top level functions
                     self.symbols.append({
                        "name": node.name,
                        "type": "function",
                        "file": rel_path,
                        "line": node.lineno
                    })

        except Exception as e:
            # Skip unparseable files
            pass
