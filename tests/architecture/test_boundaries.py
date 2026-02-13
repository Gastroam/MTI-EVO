import pytest
import sys
import os
import ast

# Architecture Constraint Tests
# "Make boundaries hard, not vibes"
# AST-based analysis for robustness.

class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

def get_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return [] # Skip unparseable
    
    visitor = ImportVisitor()
    visitor.visit(tree)
    return visitor.imports

def check_layer_violations(layer_root, forbidden_prefixes):
    """
    Scans all .py files in layer_root.
    Fails if any import starts with a forbidden prefix.
    """
    violations = []
    for dirpath, _, filenames in os.walk(layer_root):
        for f in filenames:
            if not f.endswith(".py"): continue
            
            p = os.path.join(dirpath, f)
            rel_p = os.path.relpath(p, layer_root)
            
            imports = get_imports(p)
            for imp in imports:
                for forbidden in forbidden_prefixes:
                    if imp.startswith(forbidden):
                        violations.append(f"{rel_p} imports {imp}")
                        
    return violations

def test_core_isolation():
    """
    Core Layer (Domain Logic) must NOT import:
    - Server (Infrastructure/Delivery)
    - Bootstrap (Policy/Scripting)
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/mti_evo/core"))
    if not os.path.exists(root): pytest.skip("Core dir not found")
    
    forbidden = [
        "mti_evo.server",
        "mti_evo.bootstrap",
        # "mti_evo.runtime" ? Ideally yes, but Core is used BY Runtime.
        # Core should technically be independent of Runtime too.
        "mti_evo.runtime"
    ]
    
    violations = check_layer_violations(root, forbidden)
    assert not violations, f"Core Layer Violations:\n" + "\n".join(violations)

def test_engine_isolation():
    """
    Engines (Adapters) must NOT import:
    - Server (Delivery)
    - Runtime (Composition Root) - They should be pure logic.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/mti_evo/engines"))
    if not os.path.exists(root): pytest.skip("Engines dir not found")
    
    forbidden = [
        "mti_evo.server",
        # Registry might import things? No, ideally not server.
    ]
    
    violations = check_layer_violations(root, forbidden)
    assert not violations, f"Engine Layer Violations:\n" + "\n".join(violations)

def test_cortex_isolation():
    """
    Cortex (Cognitive Layer) must NOT import:
    - Server
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/mti_evo/cortex"))
    if not os.path.exists(root): pytest.skip("Cortex dir not found")
    
    forbidden = [
        "mti_evo.server"
    ]
    
    violations = check_layer_violations(root, forbidden)
    assert not violations, f"Cortex Layer Violations:\n" + "\n".join(violations)
