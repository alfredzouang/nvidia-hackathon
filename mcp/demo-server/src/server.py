import json
import random
from mcp.server.fastmcp import FastMCP
import asyncio
# Initialize FastMCP server
server = FastMCP("demo_server")

@server.tool()
async def get_weather(location: str) -> str:
    """Get weather for a location.

    Args:
        location: Location to get weather for, e.g., city name, state, or coordinates
    
    """
    if not location:
        return "Location is required."
    
    # mock weather data
    conditions = [ "Sunny", "Rainy", "Cloudy", "Snowy" ]
    weather = {
        "location": location,
        "temperature": f"{random.randint(10, 90)}°F",
        "condition": random.choice(conditions),
    }
    return json.dumps(weather, ensure_ascii=False)

if __name__ == "__main__":
    # Start the server
    server.run(transport="sse")