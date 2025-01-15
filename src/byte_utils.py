'''
Several useful helper methods for converting between strings and bytes.
'''

from bitstring import Bits

def bitstrings_to_bytes(bitstrings: list[str]) -> bytes:
    '''
    Converts a list of 8-bit bitstrings and converts them into their
    bytes equivalent.
    
    Parameters:
        bitstrings (list[str]):
            A list of 8-bit bitstrings to convert into a sequence of bytes
    
    Returns:
        bytes:
            The byte sequence of converted bitstrings
    
    Example:
        bitstrings_to_bytes(['10100011', '11100000'])
        => b'\xa3\xe0'
    '''
    return bytes([int(bs, 2) for bs in bitstrings])

def byte_to_bitstring(b: int) -> str:
    '''
    Converts a SINGLE bytes into its bitstring equivalent.
    
    Parameters:
        b (int):
            The ONE 8-bit byte sequence to convert into a bitstring,
            represented as an int from iterating over bytes.
    
    Returns:
        str:
            The bitstring of the converted byte.
    
    Example:
        x = b'\xa2\x03'
        for b in x:
            print(byte_to_bitstring(b))
        # Prints:
        #    10100010
        #    00000011
    '''
    result: str = Bits(uint=str(b), length=8).bin
    return result