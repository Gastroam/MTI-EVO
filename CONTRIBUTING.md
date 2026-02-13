# Contributing to MTI-MOE-Bridge

Thank you for your interest in contributing to the MTI Exocortex project!

## Architecture Overview

MTI uses an 8-Lobe bio-mimetic architecture:

| Lobe | Function |
|------|----------|
| Parietal | Hardware and workspace awareness |
| Hippocampus | Episodic memory and navigation |
| Prefrontal | Executive control and planning |
| Occipital | Code vision (AST analysis) |
| Wernicke | Semantic understanding |
| Broca | Syntax generation |
| Limbic | Sentiment and style |
| Ghost | Surgical code editing |

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for extension)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Gastroam/mti-moe-bridge.git
cd mti-moe-bridge

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov black isort
```

### Running the Bridge

```bash
# Start the FastAPI bridge server
python -m src.bridge

# Or use the CLI
python -m src.cli.main
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_ghost_compression.py -v
```

## Code Style

- **Python**: Follow PEP 8, use Black formatter
- **TypeScript**: Use Prettier
- **Commits**: Use conventional commits (feat:, fix:, docs:, etc.)

```bash
# Format Python code
black src/ tests/

# Sort imports
isort src/ tests/
```

## Ghost Protocol (Token-Efficient Editing)

When contributing, prefer surgical edits over full file rewrites:

1. **Scan** the file to get Hash IDs:
   ```python
   from src.brain.lobes.ghost_indexer import GhostIndexer
   skeleton, anchors = GhostIndexer().generate_skeleton(path, content)
   ```

2. **Edit** using anchor IDs:
   ```python
   from src.brain.lobes.ghost_patcher import GhostPatcher
   result = GhostPatcher().apply(path, target_id="a8e", new_code="...")
   ```

This achieves approximately 90% token savings compared to full file operations.

## Adding New Lobes

1. Create a new file in `src/brain/lobes/`
2. Implement the lobe interface
3. Register in `src/brain/__init__.py`
4. Add tests in `tests/`

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit with conventional message
6. Push and create PR

## Reporting Issues

Please include:
- Python version
- OS (Windows/Linux/Mac)
- Steps to reproduce
- Expected vs actual behavior
- Logs if available

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built with adversity. Refined by necessity. Powered by curiosity.*
