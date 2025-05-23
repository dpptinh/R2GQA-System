# R2GQA: Retriever-Reader-Generator Question Answering System to Support Students Understanding Legal Regulations in Higher Education

## 🔍 Introduction

**R2GQA** is an automated question answering system designed to assist students in comprehending legal regulations related to higher education. It leverages a combination of modern information retrieval and natural language processing techniques to deliver accurate and user-friendly answers in Vietnamese.

The system consists of three main components working together:

1. **Retriever**
   Combines **BM25** and **SBERT** to identify and fetch the most relevant legal text passages from a large corpus based on the user’s question.

2. **Reader**
   Utilizes transformer-based models such as **XLM-RoBERTa-large** to extract precise answers from the retrieved documents.

3. **Generator**
   Synthesizes the extracted answers into natural, concise, and user-friendly responses tailored for Vietnamese legal language.


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


## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## License

This project is distributed under the CC BY-NC-SA 4.0 License for research purposes only. See [LICENSE](https://creativecommons.org/licenses/by-nc-sa/4.0/) for more information.
