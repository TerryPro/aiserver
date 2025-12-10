
import unittest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from aiserver.core.context import ContextOptimizer

class TestContextOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = ContextOptimizer(max_total_tokens=100, max_history_tokens=20)

    def test_estimate_tokens(self):
        text = "12345678"
        self.assertEqual(self.optimizer.estimate_tokens(text), 2)
        self.assertEqual(self.optimizer.estimate_tokens(""), 0)

    def test_optimize_code_context_recent_cells(self):
        # Setup: 3 cells. 
        # Cell 0 (oldest), Cell 1, Cell 2 (newest)
        # optimize_code_context receives them in order [Cell 0, Cell 1, Cell 2]
        # logic reverses them: Cell 2 (i=0), Cell 1 (i=1), Cell 0 (i=2)
        
        # Make cells small enough to fit in "recent" logic but check compression on older ones
        # But wait, optimize_code_context logic:
        # if i < 2: keep full if < 2000 chars
        # else: compress
        
        prev_codes = [
            "x = 1\n" * 10, # Cell 0 (oldest) -> Should be compressed
            "y = 2\n",      # Cell 1 -> Should be kept (i=1)
            "z = 3\n"       # Cell 2 (newest) -> Should be kept (i=0)
        ]
        
        # We need a larger max_tokens for this test to avoid the token limit cutting it off entirely
        # Cell 0 length: 60 chars -> ~15 tokens
        # Cell 1: 6 chars -> 1 token
        # Cell 2: 6 chars -> 1 token
        
        optimizer = ContextOptimizer()
        result = optimizer.optimize_code_context(prev_codes, max_tokens=100)
        
        self.assertIn("z = 3", result)
        self.assertIn("y = 2", result)
        self.assertIn("# Cell 0 (Summary)", result) # Should be compressed
        self.assertNotIn("x = 1", result) # Should be compressed summary only
        
    def test_optimize_code_context_token_limit(self):
        # Create a very long cell that exceeds max_tokens
        long_code = "a" * 400 # 100 tokens
        prev_codes = [long_code]
        
        # limit to 50 tokens
        result = optimizer = ContextOptimizer().optimize_code_context(prev_codes, max_tokens=50)
        
        # Should be empty or cut off?
        # Logic: if current_tokens + part_tokens > max_tokens: break
        # So if the first (most recent) cell is too big, it returns ""
        self.assertEqual(result, "")

    def test_compress_code(self):
        code = """
import pandas as pd
df = pd.DataFrame()
def process(data):
    '''Process data'''
    return data * 2
class Processor:
    def run(self):
        pass
"""
        # Cell index 5
        result = self.optimizer._compress_code(code, 5)
        
        self.assertIn("# Cell 5 (Summary)", result)
        self.assertIn("def process(data): ...", result)
        self.assertIn("class Processor: ...", result)
        self.assertIn("Variables: df", result)

    def test_optimize_history(self):
        # Max history tokens = 20 (from setUp)
        # Message 1: 40 chars -> 10 tokens
        # Message 2: 40 chars -> 10 tokens
        # Message 3: 40 chars -> 10 tokens
        
        msg1 = HumanMessage(content="a" * 40)
        msg2 = AIMessage(content="b" * 40)
        msg3 = HumanMessage(content="c" * 40)
        
        history = [msg1, msg2, msg3]
        
        # Reverse processing:
        # msg3 (10 tokens) -> added. Total 10.
        # msg2 (10 tokens) -> added. Total 20.
        # msg1 (10 tokens) -> 30 > 20 -> break.
        
        optimized = self.optimizer.optimize_history(history)
        
        self.assertEqual(len(optimized), 2)
        self.assertEqual(optimized[0], msg2) # Should be msg2 then msg3?
        # Wait, insert(0, msg) means we prepend.
        # 1. Process msg3. insert(0, msg3) -> [msg3]
        # 2. Process msg2. insert(0, msg2) -> [msg2, msg3]
        
        self.assertEqual(optimized[0].content, "b" * 40)
        self.assertEqual(optimized[1].content, "c" * 40)

if __name__ == '__main__':
    unittest.main()
