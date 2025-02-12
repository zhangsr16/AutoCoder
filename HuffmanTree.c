#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define MAX_SYMBOLS 256
#define MAX_CHILDREN 4
#define MAX_BUFFER_SIZE 1024

typedef struct HuffmanNode {
    int freq;
    char symbol;
    struct HuffmanNode *children[MAX_CHILDREN];
} HuffmanNode;

typedef struct {
    HuffmanNode *nodes[MAX_SYMBOLS];
    int size;
} MinHeap;

HuffmanNode* create_node(int freq, char symbol) {
    HuffmanNode *node = (HuffmanNode*)malloc(sizeof(HuffmanNode));
    node->freq = freq;
    node->symbol = symbol;
    for (int i = 0; i < MAX_CHILDREN; i++) {
        node->children[i] = NULL;
    }
    return node;
}

MinHeap* create_min_heap() {
    MinHeap *min_heap = (MinHeap*)malloc(sizeof(MinHeap));
    min_heap->size = 0;
    return min_heap;
}

void swap_nodes(HuffmanNode **a, HuffmanNode **b) {
    HuffmanNode *temp = *a;
    *a = *b;
    *b = temp;
}

void min_heapify(MinHeap *min_heap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < min_heap->size && min_heap->nodes[left]->freq < min_heap->nodes[smallest]->freq) {
        smallest = left;
    }
    if (right < min_heap->size && min_heap->nodes[right]->freq < min_heap->nodes[smallest]->freq) {
        smallest = right;
    }

    if (smallest != idx) {
        swap_nodes(&min_heap->nodes[smallest], &min_heap->nodes[idx]);
        min_heapify(min_heap, smallest);
    }
}

HuffmanNode* extract_min(MinHeap *min_heap) {
    HuffmanNode *temp = min_heap->nodes[0];
    min_heap->nodes[0] = min_heap->nodes[min_heap->size - 1];
    min_heap->size--;
    min_heapify(min_heap, 0);
    return temp;
}

void insert_min_heap(MinHeap *min_heap, HuffmanNode *node) {
    min_heap->size++;
    int i = min_heap->size - 1;
    while (i && node->freq < min_heap->nodes[(i - 1) / 2]->freq) {
        min_heap->nodes[i] = min_heap->nodes[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    min_heap->nodes[i] = node;
}

void build_min_heap(MinHeap *min_heap) {
    int n = min_heap->size - 1;
    for (int i = (n - 1) / 2; i >= 0; i--) {
        min_heapify(min_heap, i);
    }
}

MinHeap* build_huffman_tree(char data[], int freq[], int size) {
    MinHeap *min_heap = create_min_heap();
    for (int i = 0; i < size; i++) {
        min_heap->nodes[i] = create_node(freq[i], data[i]);
    }
    min_heap->size = size;
    build_min_heap(min_heap);

    while (min_heap->size != 1) {
        HuffmanNode *children[MAX_CHILDREN];
        int num_children = 0;
        for (int i = 0; i < MAX_CHILDREN && min_heap->size; i++) {
            children[num_children++] = extract_min(min_heap);
        }
        int sum_freq = 0;
        for (int i = 0; i < num_children; i++) {
            sum_freq += children[i]->freq;
        }
        HuffmanNode *internal_node = create_node(sum_freq, '\0');
        for (int i = 0; i < num_children; i++) {
            internal_node->children[i] = children[i];
        }
        insert_min_heap(min_heap, internal_node);
    }

    return min_heap;
}

void generate_codes(HuffmanNode *root, char code[], int top, char codes[MAX_SYMBOLS][MAX_SYMBOLS]) {
    if (root->children[0]) {
        for (int i = 0; i < MAX_CHILDREN; i++) {
            if (root->children[i]) {
                code[top] = '0' + i;
                generate_codes(root->children[i], code, top + 1, codes);
            }
        }
    }

    if (root->symbol != '\0') {
        code[top] = '\0';
        strcpy(codes[(unsigned char)root->symbol], code);
    }
}

void encode_assembly_code(char *assembly_code[], int num_lines, char codes[MAX_SYMBOLS][MAX_SYMBOLS], char *encoded_str) {
    encoded_str[0] = '\0';
    for (int i = 0; i < num_lines; i++) {
        char *token = strtok(assembly_code[i], " ,");
        while (token != NULL) {
            strcat(encoded_str, codes[(unsigned char)token[0]]);
            token = strtok(NULL, " ,");
        }
    }
}

void decode_huffman_code(char *encoded_str, HuffmanNode *root, char *decoded_str) {
    HuffmanNode *current_node = root;
    int idx = 0;
    for (int i = 0; encoded_str[i] != '\0'; i++) {
        int digit = encoded_str[i] - '0';
        current_node = current_node->children[digit];
        if (current_node->symbol != '\0') {
            decoded_str[idx++] = current_node->symbol;
            current_node = root;
        }
    }
    decoded_str[idx] = '\0';
}

void read_assembly_code(const char *file_path, char *assembly_code[], int *num_lines) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        fprintf(stderr, "Error opening file\n");
        exit(EXIT_FAILURE);
    }

    char line[MAX_BUFFER_SIZE];
    *num_lines = 0;
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = '\0';
        assembly_code[*num_lines] = (char *)malloc(strlen(line) + 1);
        strcpy(assembly_code[*num_lines], line);
        (*num_lines)++;
    }
    fclose(file);
}

int main() {
    const char *file_path = "assembly_code.txt";
    char *assembly_code[MAX_BUFFER_SIZE];
    int num_lines;

    read_assembly_code(file_path, assembly_code, &num_lines);

    // Split instructions into components
    char components[MAX_BUFFER_SIZE][MAX_BUFFER_SIZE];
    int component_count = 0;
    for (int i = 0; i < num_lines; i++) {
        char *token = strtok(assembly_code[i], " ,");
        while (token != NULL) {
            strcpy(components[component_count++], token);
            token = strtok(NULL, " ,");
        }
    }

    // Get frequency of each component
    int freq[MAX_SYMBOLS] = {0};
    char data[MAX_SYMBOLS];
    int unique_count = 0;
    for (int i = 0; i < component_count; i++) {
        int found = 0;
        for (int j = 0; j < unique_count; j++) {
            if (data[j] == components[i][0]) {
                freq[j]++;
                found = 1;
                break;
            }
        }
        if (!found) {
            data[unique_count] = components[i][0];
            freq[unique_count++] = 1;
        }
    }

    // Build Huffman Tree
    MinHeap *min_heap = build_huffman_tree(data, freq, unique_count);
    HuffmanNode *huffman_tree_root = min_heap->nodes[0];

    // Generate Huffman codes
    char codes[MAX_SYMBOLS][MAX_SYMBOLS] = {{0}};
    char code[MAX_SYMBOLS];
    generate_codes(huffman_tree_root, code, 0, codes);

    // Encode the assembly code
    char encoded_assembly_code[MAX_BUFFER_SIZE * MAX_SYMBOLS];
    encode_assembly_code(assembly_code, num_lines, codes, encoded_assembly_code);

    // Decode the encoded assembly code to verify
    char decoded_assembly_code[MAX_BUFFER_SIZE * MAX_SYMBOLS];
    decode_huffman_code(encoded_assembly_code, huffman_tree_root, decoded_assembly_code);

    // Print results
    printf("Assembly Code:\n");
    for (int i = 0; i < num_lines; i++) {
        printf("%s\n", assembly_code[i]);
    }
    printf("\nComponents:\n");
    for (int i = 0; i < component_count; i++) {
        printf("%s ", components[i]);
    }
    printf("\n\nFrequency of Components:\n");
    for (int i = 0; i < unique_count; i++) {
        printf("%c: %d\n", data[i], freq[i]);
    }
    printf("\nHuffman Codes:\n");
    for (int i = 0; i < unique_count; i++) {
        printf("%c: %s\n", data[i], codes[(unsigned char)data[i]]);
    }
    printf("\nEncoded Assembly Code:\n%s\n", encoded_assembly_code);
    printf("\nDecoded Components:\n%s\n", decoded_assembly_code);

    // Free memory
    for (int i = 0; i < num_lines; i++) {
        free(assembly_code[i]);
    }
    free(min_heap);

    return 0;
}
