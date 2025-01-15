from compression_utils import *
from byte_utils import *
import unittest

ETB_CHAR = "\x17"

class CompressionTests(unittest.TestCase):
    """
    Unit tests for validating the ReusableHuffman class' efficacy. Notes:
    - If this is the set of tests provided in the solution skeleton, it represents an
      incomplete set that you are expected to add to to adequately test your submission!
    - Your correctness score on the assignment will be assessed by a more complete,
      grading set of unit tests.
    - A portion of your style grade will also come from proper type hints; remember to
      validate your submission using `mypy .` and ensure that no issues are found.
    """
    
    # Constructor Tests
    # ---------------------------------------------------------------------------
    def test_constructor_t0(self) -> None:
        huff_coder = ReusableHuffman("A")
        solution = {ETB_CHAR: '0', "A": '1'}
        self.assertEqual(solution, huff_coder.get_encoding_map())
        
    def test_constructor_t1(self) -> None:
        huff_coder = ReusableHuffman("AB")
        solution = {ETB_CHAR: '10', "A": '11', 'B': '0'}
        self.assertEqual(solution, huff_coder.get_encoding_map())
        
    def test_constructor_t2(self) -> None:
        huff_coder = ReusableHuffman("ABBBCC")
        solution = {ETB_CHAR: '100', "A": '101', 'B': '0', 'C': '11'}
        self.assertEqual(solution, huff_coder.get_encoding_map())
    
    # [!] TODO: Write your own constructor tests with a larger variety of
    # characters in the corpus here!
    
    
    # Compression Tests
    # ---------------------------------------------------------------------------
    def test_compression_t0(self) -> None:
        huff_coder = ReusableHuffman("A")
        # byte 0: 1000 0000 (0 = ETB, 1 = 'A')
        # [!] Only first 2 bits of byte 0 are meaningful (rest are padding)
        compressed_message = huff_coder.compress_message("A")
        solution = bitstrings_to_bytes(['10000000'])
        self.assertEqual(solution, compressed_message)
        
    def test_compression_t1(self) -> None:
        huff_coder = ReusableHuffman("AB")
        # byte 0: 1101 0000 (10 = ETB, 11 = 'A', 0 = 'B')
        # [!] Only first 5 bits of byte 0 are meaningful (rest are padding)
        compressed_message = huff_coder.compress_message("AB")
        solution = bitstrings_to_bytes(['11010000'])
        self.assertEqual(solution, compressed_message)
        
    def test_compression_t2(self) -> None:
        huff_coder = ReusableHuffman("AB")
        # byte 0: 0111 0000 (10 = ETB, 11 = 'A', 0 = 'B')
        # [!] Only first 5 bits of byte 0 are meaningful (rest are padding)
        compressed_message = huff_coder.compress_message("BA")
        solution = bitstrings_to_bytes(['01110000'])
        self.assertEqual(solution, compressed_message)
        
    def test_compression_t3(self) -> None:
        huff_coder = ReusableHuffman("ABBBCC")
        # byte 0: 1010 0011 (100 = ETB, 101 = 'A', 0 = 'B', 11 = 'C')
        # byte 1: 1110 0000
        # [!] Only first 5 bits of byte 1 are meaningful (rest are padding)
        compressed_message = huff_coder.compress_message("ABBBCC")
        solution = bitstrings_to_bytes(['10100011', '11100000'])
        self.assertEqual(solution, compressed_message)
        
    def test_compression_t4(self) -> None:
        huff_coder = ReusableHuffman("ABBBCC")
        # byte 0: 0101 0110 (100 = ETB, 101 = 'A', 0 = 'B', 11 = 'C')
        # byte 1: 1110 0000
        # [!] Only first 5 bits of byte 1 are meaningful (rest are padding)
        compressed_message = huff_coder.compress_message("BABCBC")
        solution = bitstrings_to_bytes(['01010110', '11100000'])
        self.assertEqual(solution, compressed_message)
        
    # [!] TODO: Write your own compression tests with a greater variety of chars
    # in the corpus
    
    
    # Decompression Tests
    # ---------------------------------------------------------------------------
    
    def test_decompression_t0(self) -> None:
        huff_coder = ReusableHuffman("A")
        # byte 0: 1000 0000 (0 = ETB, 1 = 'A')
        # [!] Only first 2 bits of byte 0 are meaningful (rest are padding)
        compressed_msg: bytes = bitstrings_to_bytes(['10000000'])
        self.assertEqual("A", huff_coder.decompress(compressed_msg))
        
    def test_decompression_t1(self) -> None:
        huff_coder = ReusableHuffman("AB")
        # byte 0: 1101 0000 (10 = ETB, 11 = 'A', 0 = 'B')
        # [!] Only first 5 bits of byte 0 are meaningful (rest are padding)
        compressed_msg: bytes = bitstrings_to_bytes(['11010000'])
        self.assertEqual("AB", huff_coder.decompress(compressed_msg))
        
    def test_decompression_t2(self) -> None:
        huff_coder = ReusableHuffman("AB")
        # byte 0: 0111 0000 (10 = ETB, 11 = 'A', 0 = 'B')
        # [!] Only first 5 bits of byte 0 are meaningful (rest are padding)
        compressed_msg: bytes = bitstrings_to_bytes(['01110000'])
        self.assertEqual("BA", huff_coder.decompress(compressed_msg))
    
    def test_decompression_t3(self) -> None:
        huff_coder = ReusableHuffman("ABBBCC")
        # byte 0: 1010 0011 (100 = ETB, 101 = 'A', 0 = 'B', 11 = 'C')
        # byte 1: 1110 0000
        # [!] Only first 5 bits of byte 1 are meaningful (rest are padding)
        compressed_msg: bytes = bitstrings_to_bytes(['10100011', '11100000'])
        self.assertEqual("ABBBCC", huff_coder.decompress(compressed_msg))
        
    def test_decompression_t4(self) -> None:
        # byte 0: 0101 0110 (100 = ETB, 101 = 'A', 0 = 'B', 11 = 'C')
        # byte 1: 1110 0000
        # [!] Only first 5 bits of byte 1 are meaningful (rest are padding)
        huff_coder = ReusableHuffman("ABBBCC")
        compressed_msg: bytes = bitstrings_to_bytes(['01010110', '11100000'])
        self.assertEqual("BABCBC", huff_coder.decompress(compressed_msg))
        
    # [!] TODO: Write your own decompression tests with a greater variety of chars
    # in the corpus
        
if __name__ == '__main__':
    unittest.main()