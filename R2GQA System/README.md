# R2GQA - Automated Question Answering System

## Introduction

R2GQA is an automated question answering system developed to help students better understand legal regulations in higher education. The system combines advanced methods in information retrieval and answer generation, including:

- Searching for relevant context from legal document databases
- Extracting precise answers from question and context
- Generating natural, user-friendly responses

This project is part of research published in the paper [R2GQA: Retriever-Reader-Generator Question Answering System to Support Students Understanding Legal Regulations in Higher Education](https://arxiv.org/abs/2409.02840).

**Note**: The dataset in this repository is only permitted for research purposes, not for commercial use.

## Installation

### Requirements
- Libraries listed in `requirements.txt`

### Installation Steps

#### Clone repository:
```bash
git clone https://github.com/dpptinh/R2GQA-system
cd R2GQA
```

#### Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file with the following environment variables:
```
FULL_DATA_PATH=<path to data file>
DOCUMENTS_LINK_PATH=<path to links file>
EMBEDDING_MODEL_PATH=<path to embedding model>
EXTRACTIVE_MODEL_PATH=<path to extractive model>
ABSTRACTIVE_MODEL_PATH=<path to abstractive model>
```

## Usage

Run the application:
```bash
python main.py
```

## System Architecture

The system consists of 3 main components:

1. **Retriever**: Searches and retrieves text passages relevant to the question
2. **Reader**: Extracts precise answers from text passages
3. **Generator**: Generates natural responses based on extracted answers

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## License

This project is distributed under the CC BY-NC-SA 4.0 License for research purposes only. See [LICENSE](https://creativecommons.org/licenses/by-nc-sa/4.0/) for more information.