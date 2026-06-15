import argparse
import re
from pathlib import Path


def extract_abstracts(input_path):
    text = Path(input_path).read_text()
    blocks = re.split(r'###(\d+)\n', text)
    abstracts = {}
    for i in range(1, len(blocks), 2):
        pmid = int(blocks[i])
        sentences = re.sub(r'^[A-Z_]+\s+', '', blocks[i + 1], flags=re.MULTILINE)
        abstracts[pmid] = sentences.strip()
    return abstracts


def extract_pmids(input_path, output_path):
    text = Path(input_path).read_text()
    pmids = re.findall(r'###(\d+)', text)
    Path(output_path).write_text('\n'.join(pmids) + '\n')
    return Path(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract PMIDs from PubMed-RCT formatted text.')
    parser.add_argument('--input', required=True, help='Path to PubMed-RCT .txt file')
    parser.add_argument('--output', required=True, help='Path to write extracted PMIDs')
    args = parser.parse_args()

    out = extract_pmids(args.input, args.output)
    print(out)
