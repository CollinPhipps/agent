import ast
import operator

def safe_pow(base, exp, max_exp=1000, max_base=10_000, max_result_digits=4300):
  if not isinstance(base, (int, float)) or not isinstance(exp, (int, float)):
    raise TypeError("Exponentiation requires numeric operands")

  if abs(exp) > max_exp:
    raise ValueError(f"Exponent {exp} exceeds safety limit of {max_exp}")

  if abs(exp) > 1 and abs(base) > max_base:
    raise ValueError(f"Base {base} exceeds safety limit of {max_base}")

  if base < 0 and isinstance(exp, float) and not exp.is_integer():
    raise ValueError(
        "Negative base with fractional exponent results in complex numbers"
    )

  result = base**exp

  if isinstance(result, int):
    if len(str(abs(result))) > max_result_digits:
      raise ValueError("Exponentiation result exceeds maximum allowed digits")

  return result

class Calculator(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.operator_map = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: safe_pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        return node.value

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        op_type = type(node.op)
        if op_type not in self.operator_map:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

        return self.operator_map[op_type](left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)

        op_type = type(node.op)
        if op_type not in self.operator_map:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        
        return self.operator_map[op_type](operand)

    def generic_visit(self, node):
        raise ValueError(f"Unhandled or disallowed node type: {type(node).__name__}")

    def calculate(self, expression: str):
        try:
                tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise SyntaxError(f'{expression} is not a valid expression')

        return self.visit(tree)

if __name__ == "__main__":
    calc = Calculator()
    print(calc.calculate(expression="9**9**9**9"))