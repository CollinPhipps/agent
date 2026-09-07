import anthropic
from dotenv import load_dotenv
from agent.tools.calculator import Calculator
from agent.tools.rag import retrieve, load_database_cached, load_database_fresh
from sentence_transformers import SentenceTransformer
from vdb.store import VectorStore
import json

tools = [{
    'name': 'calculator',
    'description': 'Evaluate a basic arithmetic expression. Use this for any numeric computation instead of computing it yourself.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'expression': {'type': 'string', 'description': 'e.g. "12 * (7 + 3)"'}
        },
        'required': ['expression']
    }
    },
    {
    'name': 'rag',
    'description': ('Use this for finding answers to questions to research papers on the topics of machine learning, '
    'artificial intelligence, quantitative finance, physics, and math. Returned from this are the top k results of '
    'a search over a vector database containing abstracts of papers from all of these topcs. You are to look at the'
    'scores and metadata and determine the relevancy yourself. If nothing seems relevant, say you were unable to find anything relevent.'),
    'input_schema': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'e.g. "What are some recent papers on optimization techniques for reinforment learning?"'},
            'k': {'type': 'integer', 'description': 'Number of results to return. Defaults to 3 if not specified.'}
        },
        'required': ['query']
    }
    }
]

MAX_ITERS = 10
MODEL = 'claude-haiku-4-5'
MAX_TOKENS = 1024
SYSTEM = (
    "You are a research assistant with access to two tools: a calculator for "
    "arithmetic, and a search tool over a database of research paper abstracts "
    "(machine learning, math, physics, and quantitative finance). "
    "Use the calculator instead of doing arithmetic yourself. "
    "Use the search tool for questions about research topics or papers instead "
    "of answering from memory. "
    "When you get search results back, cite the specific paper titles you're "
    "drawing from. If the search tool returns no relevant results, say so "
    "plainly rather than guessing or making something up."
)

def run_turn(messages, client, calculator, embed_model, vdb):
    for _ in range(MAX_ITERS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=tools,
            messages=messages
        )
        messages.append({'role': 'assistant', 'content': response.content})

        if response.stop_reason != 'tool_use':
            break

        tool_results = []
        for block in response.content:
            if block.type == 'tool_use':
                error = False
                error_result = {'type': 'tool_result', 'tool_use_id': block.id, 'content': '_', 'is_error': True}
                if block.name == 'calculator':
                    try:
                        output = calculator.calculate(block.input['expression'])
                    except Exception as e:
                        error_result['content'] = str(e)
                        error = True
                elif block.name == 'rag':
                    k = block.input.get('k', 3)
                    k = min(k, 10)
                    try:
                        output = retrieve(vdb, embed_model, block.input['query'], k)
                    except Exception as e:
                        error_result['content'] = str(e)
                        error = True
                if error:
                    tool_results.append(error_result)
                else:
                    tool_results.append({'type': 'tool_result', 'tool_use_id': block.id, 'content': json.dumps(output)})

        messages.append({'role': 'user', 'content': tool_results})

    else:
        raise RuntimeError(f"hit {MAX_ITERS} iterations without a final answer")

def main():
    print("Initializing...")
    load_dotenv()
    calculator = Calculator()
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    vdb = VectorStore(dim=384)
    load_database_cached(vdb, 'embed.npy', 'metadata.json')
    client = anthropic.Anthropic()

    messages = []
    while True:
        user_input = input('You: ')
        print('\n----------------\n')
        if user_input in ('quit', 'exit'): break
        messages.append({'role': 'user', 'content': user_input})
        run_turn(messages, client, calculator, embed_model, vdb)
        final_text = "".join(block.text for block in messages[-1]['content'] if block.type == 'text')
        print(final_text)
        print('\n----------------\n')


if __name__ == "__main__":
    main()