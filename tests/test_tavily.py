import pytest
import asyncio
import sys
import os

# Add the project root to sys.path so we can import agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.tools import web_search

@pytest.mark.asyncio
async def test_tavily_search():
    """
    Test the web_search tool that uses Tavily API.
    Run this with: pytest tests/test_tavily.py -s
    """
    query = "Latest advancements in AI"
    print(f"\n--- Testing Tavily Search with query: '{query}' ---")
    
    result = await web_search(query)
    
    print("\n--- Search Results ---")
    print(result)
    print("----------------------")
    
    # Assert that we didn't just get an error string
    assert "Error:" not in result, f"Search failed with error: {result}"
    
    # We should get some content if the API key is valid
    if result == "No results found.":
        print("\nNote: No results found. This might be fine depending on the query.")
    elif "Search result placeholder for:" in result:
        pytest.fail("Still using the placeholder implementation!")
    
    # Check that it's a string
    assert isinstance(result, str)

if __name__ == "__main__":
    asyncio.run(test_tavily_search())
