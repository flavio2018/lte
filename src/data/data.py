import numpy as np


class Addition:
    def __init__(self):
        self.operands = 2
        
    def evaluate(self, values):
        op1, op2 = values
        return op1 + op2
    
    def generate_code(self, codes):
        code_op1, code_op2 = codes
        return "(" + str(code_op1) + "+" + str(code_op2) + ")"


class Subtraction:
    def __init__(self):
        self.operands = 2
    
    def evaluate(self, values):
        op1, op2 = values
        return op1 - op2

    def generate_code(self, codes):
        code_op1, code_op2 = codes
        return "(" + str(code_op1) + "-" + str(code_op2) + ")"


class Multiplication:
    def __init__(self):
        self.operands = 2
        self.used_params = 0
    
    def evaluate(self, values):
        op1, op2 = values
        return op1 * op2

    def generate_code(self, codes):
        code_op1, code_op2 = codes
        return "(" + str(code_op1) + "*" + str(code_op2) + ")"


class Assignment:
    def __init__(self, used_letters):
        self.operands = 1
        self.used_letters = used_letters
    
    def evaluate(self, values):
        return values[0]
    
    def generate_code(self, codes):
        letter = np.random.randint(len(self.used_letters))
        letter_code = self.used_letters.pop(letter)
        valueless_code = letter_code + '=' + str(codes[0])
        return letter_code, valueless_code


class IfStatement:
    def __init__(self):
        self.operands = 4
        self.geq = True
        
    def evaluate(self, values):
        res1, res2, op1, op2 = values
        self.geq = True if np.random.rand() > 0.5 else False
        if self.geq:
            if op1 > op2:
                return res1
            else:
                return res2
        else:
            if op1 < op2:
                return res1
            else:
                return res2
        
    def generate_code(self, codes):
        code_res1, code_res2, code_op1, code_op2 = codes
        op = ">" if self.geq else "<"
        return "(" + code_res1 + "if" + code_op1 + op + code_op2 + "else" + code_res2 + ")"


class ForLoop:
    def __init__(self, length):
        self.operands = 1
        self.length = length
        
    def _set_accumulator(self):
        self.accumulator_code = 'x'
        self.accumulator_value = np.random.randint(10**self.length)
        
    def evaluate(self, values):
        self._set_accumulator()
        self.num_loops = np.random.randint(1, 10)
        accumulator = values[0] 
        for l in range(self.num_loops):
            accumulator += self.accumulator_value
        return accumulator
    
    def generate_code(self, codes):
        return "(x=" + codes[0] + "for[" + str(self.num_loops) + "]" + "x+=" + str(self.accumulator_value) + ")"


def generate_sample(length, nesting, split='train', ops='asmif'):
    program_split = ''
    
    while(program_split != split):
        stack = []
        ops_dict = {
            "a": Addition(),
            "s": Subtraction(),
            "m": Multiplication(),
            "i": IfStatement(),
            "f": ForLoop(length),
        }
        # ops = [Addition(), Subtraction(), Multiplication()] #, Assignment(used_letters), IfStatement(), ForLoop(used_letters, length)]
        ops_subset = [v for k, v in ops_dict.items() if k in ops]
        program = ''
        
        for i in range(nesting):
            op = ops_subset[np.random.randint(len(ops_subset))]
            values = []
            codes = []

            for param in range(op.operands):
                if stack:
                    value, code = stack.pop()
                else:
                    if isinstance(op, Multiplication) and op.used_params == 0:  # for the first parameter of multiplication
                        value = np.random.randint(4*(length-1), 4*length)
                        op.used_params += 1
                    else:
                        value = np.random.randint(10**(length-1), 10**length)
                    code = str(value)
                values.append(value)
                codes.append(code)
            new_value = op.evaluate(values)
            new_code = op.generate_code(codes)
            stack.append((new_value, new_code))
        final_value, final_code = stack.pop()
        program += final_code[1:-1]

        program_hash = hash(program)
        if program_hash % 3 == 0:
            program_split = 'train'
        elif program_hash % 3 == 1:
            program_split = 'valid'
        else:
            program_split = 'test'

    return program, str(final_value)
