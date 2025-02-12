import heapq
from collections import defaultdict, Counter

class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(symbols_with_freqs):
    heap = [HuffmanNode(freq, symbol) for symbol, freq in symbols_with_freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0] if heap else None

def generate_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}

    if node.symbol is not None:
        codebook[node.symbol] = prefix

    if node.left:
        generate_huffman_codes(node.left, prefix + "0", codebook)
    if node.right:
        generate_huffman_codes(node.right, prefix + "1", codebook)

    return codebook

def encode_assembly_code(assembly_code, codebook):
    encoded_str = ""
    for instruction in assembly_code:
        for component in instruction.split():
            encoded_str += codebook[component]
    return encoded_str

def decode_huffman_code(encoded_str, root):
    decoded_symbols = []
    current_node = root
    for bit in encoded_str:
        current_node = current_node.left if bit == "0" else current_node.right
        if current_node.symbol is not None:
            decoded_symbols.append(current_node.symbol)
            current_node = root

    return decoded_symbols

def read_assembly_code(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]

def main():
    # Example assembly code lines
    assembly_code = read_assembly_code("assembly_code.txt")

    # Split instructions into components
    components = []
    for instruction in assembly_code:
        components.extend(instruction.split())

    # Get frequency of each component
    freqs = Counter(components)

    # Build Huffman Tree
    huffman_tree_root = build_huffman_tree(freqs)

    # Generate Huffman codes
    huffman_codes = generate_huffman_codes(huffman_tree_root)

    # Encode the assembly code
    encoded_assembly_code = encode_assembly_code(assembly_code, huffman_codes)

    # Decode the encoded assembly code to verify
    decoded_assembly_code = decode_huffman_code(encoded_assembly_code, huffman_tree_root)

    # Print results
    print("Assembly Code:", assembly_code)
    print("Components:", components)
    print("Frequency of Components:", freqs)
    print("Huffman Codes:", huffman_codes)
    print("Encoded Assembly Code:", encoded_assembly_code)
    print("Decoded Components:", decoded_assembly_code)

if __name__ == "__main__":
    main()
