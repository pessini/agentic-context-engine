# Starter Templates

Copy-paste starting points for ACE with different LLM providers.

## Templates

### [quickstart_litellm.py](quickstart_litellm.py)
**LiteLLM integration** (100+ providers: OpenAI, Anthropic, Google, etc.)

- Multiple provider examples (OpenAI, Anthropic, Google, Azure, Groq)
- Fallback configuration
- Retry strategies
- Complete offline training example

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python quickstart_litellm.py
```

### [langchain_starter_template.py](langchain_starter_template.py)
**LangChain integration** with ACE learning

- Chain creation with prompts
- Agent with tools
- Integration patterns
- Async support

**Requirements:**
```bash
pip install ace-framework[langchain]
export OPENAI_API_KEY="your-key"
python langchain_starter_template.py
```

### [ollama_starter_template.py](ollama_starter_template.py)
**Local Ollama models** with ACE

- Local model setup (llama2, mistral, etc.)
- No API key required
- Offline learning

**Requirements:**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Run: `python ollama_starter_template.py`

## How to Use

1. **Pick a template** matching your LLM provider
2. **Copy the file** as your starting point
3. **Customize** samples and environment for your use case
4. **Run** and iterate

## See Also

- [Main Examples](../) - All ACE examples
- [Quick Start Guide](../../docs/QUICK_START.md) - Step-by-step tutorial
- [API Reference](../../docs/API_REFERENCE.md) - Complete API docs
