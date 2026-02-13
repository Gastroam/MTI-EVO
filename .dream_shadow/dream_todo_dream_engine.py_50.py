# Dream: source: str  # e.g., "TODO", "design_doc", "error_log"
# Generated: 2026-01-28T15:37:06.312591

from typing import Dict, Any
import time

class Scenario:
    """A scenario to dream about."""

    id: str
    description: str
    source: str  # e.g., "TODO", "design_doc", "error_log"
    priority: float  # 0-1, higher = more important to dream about
    context: Dict[str, Any] = {}
    created_at: float = time.time()

    def __init__(self, id: str, description: str, source: str, priority: float = 0.5):
        """
        Initializes a Scenario object.

        Args:
            id: Unique identifier for the scenario.
            description: A brief description of the scenario.
            source: The origin of the scenario (e.g., "TODO", "design_doc").
            priority: The priority of the scenario (0-1).
        """
        self.id = id
        self.description = description
        self.source = source
        self.priority = priority

    def __repr__(self):
        return (f"Scenario(id={self.id}, description='{self.description}', "
                f"source='{self.source}', priority={self.priority}, context={self.context}, "
                f"created_at={self.created_at})")

    def update_context(self, key: str, value: Any):
        """
        Updates the context dictionary with the given key-value pair.

        Args:
            key: The key to update.
            value: The value to set.
        """
        self.context[key] = value

    def validate(self):
        """
        Validates the scenario data.

        Returns:
            True if the scenario data is valid, False otherwise.
        """
        if not isinstance(self.id, str):
            print("Error: ID must be a string.")
            return False
        if not isinstance(self.description, str):
            print("Error: Description must be a string.")
            return False
        if not isinstance(self.source, str):
            print("Error: Source must be a string.")
            return False
        if not (0 <= self.priority <= 1):
            print("Error: Priority must be between 0 and 1.")
            return False
        return True

if __name__ == '__main__':
    # Example usage
    scenario1 = Scenario(id="123", description="Implement feature X", source="TODO")
    print(scenario1)
    scenario1.update_context("additional_info", "This is some extra information")
    print(scenario1)

    scenario2 = Scenario(id="456", description="Fix bug Y", source="error_log", priority=0.9)
    print(scenario2)

    if scenario2.validate():
        print("Scenario 2 is valid.")
    else:
        print("Scenario 2 is invalid.")