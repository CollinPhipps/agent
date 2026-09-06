import anthropic
from dotenv import load_dotenv
from agent.tools.calculator import Calculator

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
]

MAX_ITERS = 10
MODEL = 'claude-haiku-4-5'
MAX_TOKENS = 1024
SYSTEM = 'You are a helpful assistant with tools...'

def dispatch_tool(name, input):
    if name == 'calculator':
        calc = Calculator()
        return calc.calculate(input['expression'])

def main():
    load_dotenv()

    client = anthropic.Anthropic()
    messages = [{'role': 'user', 'content': 'add 27.5 to 76, then divide by 2.'}]

    for _ in range(MAX_ITERS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=tools,
            messages=messages
        )

        print(response.content)
        print(response.stop_reason)

        messages.append({'role': 'assistant', 'content': response.content})

        if response.stop_reason != 'tool_use':
            break

        tool_results = []
        for block in response.content:
            if block.type == 'tool_use':
                print(f"BLOCK NAME: {block.name}, BLOCK INPUT: {block.input}")
                output = dispatch_tool(block.name, block.input)
                tool_results.append({'type': 'tool_result', 'tool_use_id': block.id, 'content': str(output)})

        messages.append({'role': 'user', 'content': tool_results})

    else:
        raise RuntimeError(f"hit {MAX_ITERS} iterations without a final answer")

if __name__ == "__main__":
    main()