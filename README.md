# Building Scalable MCP Servers with Domain-Driven Design

This repository demonstrates how to apply Domain-Driven Design (DDD) principles to build maintainable, scalable Model Context Protocol (MCP) servers. It accompanies the blog post [Building Scalable MCP Servers with Domain-Driven Design](https://medium.com/@chris.p.hughes10/building-scalable-mcp-servers-with-domain-driven-design-fb9454d4c726?source=friends_link&sk=9edabaa1479c98edbbe3ce969c675664).

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- An API key for either:
  - Azure OpenAI (recommended)
  - Anthropic Claude

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp-explore
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Configure your API keys:
```bash
# For Azure OpenAI (copy and fill in your details)
cp env/azure.env.example env/azure.env

# For Anthropic (copy and fill in your API key)  
cp env/anthropic.env.example env/anthropic.env
```

### Try the Examples

**Start the domain-driven weather server:**
```bash
uv run python src/run_weather_service.py
```

**Use a client to interact with the server:**
This does not require the server to already be running.

```bash
# Simple client with basic tool calling
uv run src/chat.py src/run_weather_service.py azure simple

# Intelligent resource selector (automatically loads relevant historical data)
uv run src/chat.py src/run_weather_service.py azure resource_selector

# Prompt-aware client (recognizes structured requests)
uv run src/chat.py src/run_weather_service.py anthropic prompt_aware
```

**Try these example queries:**
- "What's the weather forecast for San Francisco?" (37.7749, -122.4194)
- "Are there any weather alerts in California?"
- "Show me recent weather patterns for Texas based on historical alerts"
- "I need a comprehensive weather analysis for Los Angeles"

## Repository Structure

```
├── blog/                          # Blog post and documentation
│   ├── blog_post.md              # Main tutorial
│   └── blog_post.ipynb           # Jupyter notebook version
├── src/
│   ├── client/                   # MCP client implementations
│   │   ├── mcp_client.py        # Three client strategies
│   │   └── providers/           # LLM provider abstractions
│   ├── servers/
│   │   ├── simple_weather/      # Basic MCP server (before DDD)
│   │   └── weather_service/     # Domain-driven implementation
│   │       ├── domain/          # Domain models, services, repositories
│   │       ├── infrastructure/  # HTTP clients, external concerns
│   │       └── application/     # MCP server application service
│   ├── chat.py                  # Interactive client runner
│   └── run_weather_service.py   # Domain-driven server runner
├── env/                         # Environment configuration templates
├── pyproject.toml              # uv project configuration
└── uv.lock                     # uv dependency lock file
```

## Architecture Overview

This repository demonstrates the evolution from a simple, tightly-coupled MCP server to a sophisticated domain-driven architecture:

### Simple Implementation
- Direct API calls mixed with business logic
- Hard to test without external dependencies
- Tightly coupled to MCP infrastructure
- Located in: `src/servers/simple_weather/`

### Domain-Driven Implementation
- **Domain Layer**: Pure business logic (`WeatherService`, `Forecast`, `WeatherAlert`)
- **Application Layer**: MCP protocol orchestration (`WeatherMCPService`)
- **Infrastructure Layer**: External concerns (HTTP clients, file storage)
- Located in: `src/servers/weather_service/`

## MCP Server Features

The domain-driven weather server exposes:

### Tools (LLM can call directly)
- `get_forecast(latitude, longitude)` - Real-time weather forecasts
- `get_alerts(state)` - Current weather alerts for US states

### Resources (LLM can read for context)
- `historical://alerts/{state}` - Historical alert data for analysis

### Prompts (Structured workflow templates)
- `weather_analysis_prompt(location)` - Comprehensive weather analysis template

## Client Implementations

Three different client strategies showcase MCP's flexibility:

### 1. SimpleChatMCPClient
Basic tool calling and conversation management. Good for understanding core MCP concepts.

### 2. LLMResourceSelector  
Uses AI to automatically select relevant resources based on user queries. Demonstrates intelligent context management.

### 3. PromptAwareMCPClient
Recognizes when users want structured workflows and applies appropriate prompt templates.

## Learning Path

1. **Start with the blog post**: [Building Scalable MCP Servers with Domain-Driven Design](https://medium.com/@chris.p.hughes10/building-scalable-mcp-servers-with-domain-driven-design-fb9454d4c726?source=friends_link&sk=9edabaa1479c98edbbe3ce969c675664)

2. **Compare implementations**: 
   - Simple: `src/servers/simple_weather/weather.py`
   - Domain-driven: `src/servers/weather_service/`

3. **Try different clients**: Run the examples above to see how different client strategies work

4. **Explore the domain model**: Look at `src/servers/weather_service/domain/models.py` to see how weather concepts are modeled

5. **Understand the architecture**: Follow the separation between domain, application, and infrastructure layers

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Common commands:

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Run a script
uv run src/run_weather_service.py

# Format code (if you have ruff configured)
uv run ruff format .

# Check code quality
uv run ruff check .
```

## 📄 License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built using [FastMCP](https://github.com/jlowin/fastmcp) for rapid MCP server development
- Weather data provided by the [National Weather Service API](https://www.weather.gov/documentation/services-web-api)
- Inspired by Domain-Driven Design principles from Eric Evans and Martin Fowler
