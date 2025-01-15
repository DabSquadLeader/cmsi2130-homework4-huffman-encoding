import copy
from queue import *
from dataclasses import *
from typing import *
from byte_utils import *

# [!] Important: This is the character code of the End Transmission Block (ETB)
# Character -- use this constant to signal the end of a message
ETB_CHAR = "\x17"

class HuffmanNode:
    '''
    HuffmanNode class to be used in construction of the Huffman Trie
    employed by the ReusableHuffman encoder/decoder below.
    '''
    
    # Educational Note: traditional constructor rather than dataclass because of need
    # to set default values for children parameters
    def __init__(self, char: str, freq: int, 
                 zero_child: Optional["HuffmanNode"] = None, 
                 one_child: Optional["HuffmanNode"] = None):
        '''
        HuffNodes represent nodes in the HuffmanTrie used to create a lossless
        encoding map used for compression. Their properties are given in this
        constructor's arguments:
        
        Parameters:
            char (str):
                Really, a single character, storing the character represented
                by a leaf node in the trie
            freq (int):
                The frequency with which the character / characters in a subtree
                appear in the corpus
            zero_child, one_child (Optional[HuffmanNode]):
                The children of any non-leaf, or None if a leaf; the zero_child
                will always pertain to the 0 bit part of the prefix, and vice
                versa for the one_child (which will add a 1 bit to the prefix)
        '''
        self.char = char
        self.freq = freq
        self.zero_child = zero_child
        self.one_child = one_child

    def is_leaf(self) -> bool:
        '''
        Returns:
            bool:
                Whether or not the current node is a leaf
        '''
        return self.zero_child is None and self.one_child is None
    
    def __lt__(self, other: "HuffmanNode") -> bool:
        if self.freq == other.freq:
            return self.char < other.char
        return self.freq < other.freq

    def __eq__(self, other: Any) -> bool:
        return bool(self.freq == other.freq)

class ReusableHuffman:
    '''
    ReusableHuffman encoder / decoder that is trained on some original
    corpus of text and can then be used to compress / decompress other
    text messages that have similar distributions of characters.
    '''
    
    def __init__(self, corpus: str):
        '''
        Constructor for a new ReusableHuffman encoder / decoder that is fit to
        the given text corpus and can then be used to compress and decompress
        messages with a similar distribution of characters.
        
        Parameters:
            corpus (str):
                The text corpus on which to fit the ReusableHuffman instance,
                which will be used to construct the encoding map
        '''
        self._encoding_map: dict[str, str] = dict()

        chars = set(corpus)
        self.leaves: PriorityQueue[HuffmanNode] = PriorityQueue()
        
        etb_node = HuffmanNode(ETB_CHAR, 1)
        self.leaves.put(etb_node)
        
        if len(corpus) > 0:
            # Create initial nodes
            for item in chars:
                letter_count = corpus.count(item)
                node = HuffmanNode(item, letter_count)
                self.leaves.put(node)
                
            # Create trie using leaves and find root
            self._trie_root = self.grow_trie(self.leaves)
            self._encoding_map = self.create_encoding_map(self._trie_root, "")
            
        else:
            self._encoding_map = {ETB_CHAR: '0'}
        
        
    def grow_trie(self, trie: PriorityQueue) -> Any:
        # >> [WN] Provide proper docstrings for ALL methods, including helpers you write (-1)

        # Repeat until priority queue only has root
        while trie.qsize() > 1:
            zero_item = trie.get()
            one_item = trie.get()
            
            # Tiebreaking
            # >> [WN] Lots of code repetition below: notice how you have the same HuffNode parent declared
            # in each and then setting the children -- this should've been handled in your HuffmanNode's
            # __lt__ method to avoid precisely this (-1)
            if zero_item.char > one_item.char:
                new_char = one_item.char
            else:
                new_char = zero_item.char
            node = HuffmanNode(new_char, zero_item.freq + one_item.freq, zero_item, one_item)
            trie.put(node)
            
        return trie.get()
        
    def create_encoding_map(self, node: Optional[HuffmanNode], byte: str) -> Dict[str, str]:
        # >> [WN] Provide proper docstrings for ALL methods, including helpers you write

        # Uses recursion to form encoding map
        encoding_map = {}
        if node is not None:
            if node.is_leaf():
                encoding_map[node.char] = byte
            else:
                encoding_map.update(self.create_encoding_map(node.zero_child, byte + "0"))
                encoding_map.update(self.create_encoding_map(node.one_child, byte + "1"))
        return encoding_map                
    
    def get_encoding_map(self) -> dict[str, str]:
        '''
        Simple getter for the encoding map that, after the constructor is run,
        will be a dictionary of character keys mapping to their compressed
        bitstrings in this ReusableHuffman instance's encoding
        
        Example:
            {ETB_CHAR: 10, "A": 11, "B": 0}
            (see unit tests for more examples)
        
        Returns:
            dict[str, str]:
                A copy of this ReusableHuffman instance's encoding map
        '''
        return copy.deepcopy(self._encoding_map)
    
    # Compression
    # ---------------------------------------------------------------------------
    
    def compress_message(self, message: str) -> bytes:
        '''
        Compresses the given String message / text corpus into its Huffman-coded
        bitstring, and then converted into a Python bytes type.
        
        [!] Uses the _encoding_map attribute generated during construction.
        
        Parameters:
            message (str):
                String representing the corpus to compress
        
        Returns:
            bytes:
                Bytes storing the compressed corpus with the Huffman coded
                bytecode. Formatted as (1) the compressed message bytes themselves,
                (2) terminated by the ETB_CHAR, and (3) [Optional] padding of 0
                bits to ensure the final byte is 8 bits total.
        
        Example:
            huff_coder = ReusableHuffman("ABBBCC")
            compressed_message = huff_coder.compress_message("ABBBCC")
            # [!] Only first 5 bits of byte 1 are meaningful (rest are padding)
            # byte 0: 1010 0011 (100 = ETB, 101 = 'A', 0 = 'B', 11 = 'C')
            # byte 1: 1110 0000
            solution = bitstrings_to_bytes(['10100011', '11100000'])
            self.assertEqual(solution, compressed_message)
        '''
        encode_map: dict[str, str] = self.get_encoding_map()
        
        # Manually add ETB for compression
        message += ETB_CHAR
        bitstr: str = ''
        final_bitstr: list[str] = []
        for char in message:
            if char in encode_map:
                # Add bits to bitstring until leaf is found
                for num in encode_map[char]:
                    # When bitstring is greater than 8 bytes, add to final bitstring and reset
                    if len(bitstr) == 8:
                        final_bitstr.append(bitstr)
                        bitstr = ''
                    bitstr += num
        # Manually add padding
        if len(bitstr) < 8:
            while len(bitstr) != 8:
                bitstr += '0'
            final_bitstr.append(bitstr)

        # >> [WN] Bit uneccessary to have a variable compressed_msg. You can combine the
        # following two lines together and they will make your code more simplified.        
        compressed_msg = bitstrings_to_bytes(final_bitstr)
        return compressed_msg

    # Decompression
    # ---------------------------------------------------------------------------
    
    def decompress(self, compressed_msg: bytes) -> str:
        '''
        Decompresses the given bytes representing a compressed corpus into their
        original character format.
        
        [!] Should use the Huffman Trie generated during construction.
        
        Parameters:
            compressed_msg (bytes):
                Formatted as (1) the compressed message bytes themselves,
                (2) terminated by the ETB_CHAR, and (3) [Optional] padding of 0
                bits to ensure the final byte is 8 bits total.
        
        Returns:
            str:
                The decompressed message as a string.
        
        Example:
            huff_coder = ReusableHuffman("ABBBCC")
            # byte 0: 1010 0011 (100 = ETB, 101 = 'A', 0 = 'B', 11 = 'C')
            # byte 1: 1110 0000
            # [!] Only first 5 bits of byte 1 are meaningful (rest are padding)
            compressed_msg: bytes = bitstrings_to_bytes(['10100011', '11100000'])
            self.assertEqual("ABBBCC", huff_coder.decompress(compressed_msg))
        '''
        encoded_msg: str = ''
        decoded_msg: str = ''

        for byte in compressed_msg:
            encoded_msg += byte_to_bitstring(byte)
        
        trie_root: "HuffmanNode" = self._trie_root
        node = trie_root
        
        # Traverse trie using 0 and 1 paths until leaf is found
        for bit in encoded_msg: 
            if bit == '0' and node.zero_child is not None:
                node = node.zero_child
            elif bit == '1' and node.one_child is not None:
                node = node.one_child

            if node.is_leaf():
                # Return message once ETB is found
                if node.char == ETB_CHAR:
                    return decoded_msg
                decoded_msg += node.char
                node = self._trie_root
    
        return decoded_msg
    
# ===================================================
# >>> [WN] Summary
# Great submission that has a ton to like and was
# obviously well-tested apart from a few edge cases
# that managed to slip by. Generally clean style (apart
# from a few quibbles above), and I think that with 
# more time debugging, this code will be golden. Keep 
# up the good effort!
# ---------------------------------------------------
# >>> [WN] Style Checklist
# [X] = Good, [~] = Mixed bag, [ ] = Needs improvement
# 
# [X] Variables and helper methods named and used well
# [X] Proper and consistent indentation and spacing
# [X] Proper docstrings provided for ALL methods
# [X] Logic is adequately simplified
# [X] Code repetition is kept to a minimum
# ---------------------------------------------------
# Correctness:        91.0 / 100 (-1.5 / missed test)
# Style Penalty:      -2.0
# Mypy Compliance:    -0.0       (-5 if mypy unhappy)
# Total:              89.0 / 100
# ===================================================