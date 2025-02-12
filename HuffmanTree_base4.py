import heapq
from collections import defaultdict, Counter

class HuffmanNode:
    def __init__(self, freq, symbol=None, children=None):
        self.freq = freq
        self.symbol = symbol
        self.children = children if children else [None] * 4

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(symbols_with_freqs):
    heap = [HuffmanNode(freq, symbol) for symbol, freq in symbols_with_freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        children = [heapq.heappop(heap) for _ in range(4) if heap]
        merged_freq = sum(child.freq for child in children)
        merged_node = HuffmanNode(merged_freq, children=children)
        heapq.heappush(heap, merged_node)

    return heap[0] if heap else None

def generate_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}

    if node.symbol is not None:
        codebook[node.symbol] = prefix

    for i, child in enumerate(node.children):
        if child:
            generate_huffman_codes(child, prefix + str(i), codebook)

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
    for digit in encoded_str:
        current_node = current_node.children[int(digit)]
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
