
import unittest
import dream_interface as dream_api # Functional Binding: Testing the Real Logic
from rich.console import Console

class TestDreamAPI(unittest.TestCase):
    def setUp(self):
        self.console = Console()
        self.console.print("\n[bold magenta]🧪 Testing Dream API (Autopoiesis Verification)[/]")

    def test_get_response(self):
        response = dream_api.get_response("Hello")
        self.assertIn("Echo", response)
        self.console.print(f"[green]✔ get_response('Hello') -> {response}[/]")

    def test_get_options(self):
        options = dream_api.get_options()
        self.assertIsInstance(options, list)
        self.console.print(f"[green]✔ get_options() -> {options}[/]")

    def test_synaptic_projector(self):
        result = dream_api.synaptic_projector()
        self.assertIn("Synapses projected", result) # Flexible match
        self.console.print(f"[green]✔ synaptic_projector() -> {result}[/]")
        
    def test_confidence_score(self):
        score = dream_api.confidence_score()
        self.assertGreater(score, 0.8) # Dynamic score check
        self.console.print(f"[green]✔ confidence_score() -> {score}[/]")

    def test_dream_engine(self):
        engine = dream_api.get_dream_engine()
        # "Oneiric Analyzer" is the formal name for the Dream Engine
        is_valid = "Dream" in engine or "Oneiric" in engine
        self.assertTrue(is_valid, f"Unexpected Engine Name: {engine}")
        self.console.print(f"[green]✔ get_dream_engine() -> {engine}[/]")

if __name__ == '__main__':
    unittest.main()
