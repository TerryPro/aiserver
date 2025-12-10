import unittest
from aiserver.utils.code_analysis import extract_code_metadata

class TestCodeAnalysis(unittest.TestCase):
    def test_extract_variables(self):
        code = """
x = 1
y = 2
"""
        metadata = extract_code_metadata(code)
        self.assertEqual(len(metadata["variables"]), 2)
        self.assertEqual(metadata["variables"][0]["name"], "x")
        self.assertEqual(metadata["variables"][1]["name"], "y")

    def test_extract_functions(self):
        code = """
def my_func(a, b):
    '''Docstring'''
    return a + b
"""
        metadata = extract_code_metadata(code)
        self.assertEqual(len(metadata["functions"]), 1)
        self.assertEqual(metadata["functions"][0]["name"], "my_func")
        self.assertEqual(metadata["functions"][0]["args"], ["a", "b"])
        self.assertEqual(metadata["functions"][0]["doc"], "Docstring")

    def test_syntax_error(self):
        code = "def broken_func("
        metadata = extract_code_metadata(code)
        self.assertEqual(len(metadata["functions"]), 0)

if __name__ == '__main__':
    unittest.main()
