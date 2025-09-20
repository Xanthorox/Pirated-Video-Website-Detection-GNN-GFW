# Pirated Video Website Detection using Heterogeneous Graph Networks

This project is a deep dive into detecting pirated video websites using Graph Neural Networks (GNNs). The core idea is that illicit websites, while appearing distinct, often share underlying infrastructure (like IP addresses, SSL certificates, or WHOIS details). By modeling these complex relationships as a heterogeneous graph, we can train a powerful model to identify these sites more effectively than by looking at their content alone.

This repository contains the complete code from my research and experimentation, all within a single Jupyter Notebook (`graph.ipynb`).

## Key Features

*   **Heterogeneous Graph Modeling**: The system models the ecosystem as a heterogeneous graph with multiple node types (websites, third-party services, IPs, certificates, registrants, DNS servers, emails) and the relationships between them.
*   **State-of-the-Art GNN Architectures**: I've implemented and compared several powerful GNN models suitable for this task:
    *   **Heterogeneous Graph Transformer (HGT)**
    *   **Heterogeneous Attention Network (HAN)**
    *   **Relational Graph Convolutional Network (RGCN)**
*   **Advanced Loss Function with ReNode**: To improve performance, this project implements the ReNode algorithm. It uses a PageRank-derived metric to re-weight the training loss, forcing the model to focus on more difficult or important examples.
*   **Rich Feature Engineering**: Node features are generated using techniques like `chars2vec` to create meaningful vector representations from textual data like domain names.

## How It Works

The entire pipeline is contained within `graph.ipynb`:

1.  **Data Loading**: Raw data about website relationships and labels is loaded from CSV files.
2.  **Graph Construction**: A heterogeneous graph is built using the **Deep Graph Library (DGL)**, capturing the entities as nodes and their connections as edges.
3.  **Feature Generation**: Initial features for all nodes in the graph are created.
4.  **Model Training**: A GNN model is trained to perform node classification, predicting whether a website node is "pirated" or "legitimate".
5.  **Evaluation**: The model's performance is evaluated using standard metrics like Precision, Recall, and F1-Score, complete with classification reports and confusion matrices.

## Repository Structure

```
├── graph.ipynb             # The main Jupyter Notebook with all code.
├── README.md               # You are here!
├── requirements.txt        # Python dependencies.
└── dataset/
    ├── graph-data.zip      # Compressed raw data.
    └── graph-data/         # Processed heterogeneous graph data (.pkl files).
```

## Getting Started

### Prerequisites

You'll need Python 3 and the libraries listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/PiratedVideoWebsite-GNN.git
    cd PiratedVideoWebsite-GNN
    ```

2.  **Install dependencies:**
    I recommend setting up a virtual environment first.
    ```bash
    pip install -r requirements.txt
    ```
    *Note on DGL:* The `dgl` library often requires a specific build depending on your CUDA version. The `requirements.txt` file lists the standard `dgl`, but you may need to visit the [DGL installation page](https://www.dgl.ai/pages/start.html) for a command tailored to your system for full GPU support.

3.  **Set up the data:**
    Unzip the graph data:
    ```bash
    unzip dataset/graph-data.zip -d dataset/
    ```

4.  **Run the notebook:**
    Launch Jupyter and open `graph.ipynb` to explore the code, run the models, and see the results.
    ```bash
    jupyter notebook
    ```

## Future Work

This project was a great learning experience and a solid proof-of-concept. Here are some ways it could be improved:

*   **Experiment with more GNN architectures.**
*   **Incorporate more diverse node features** (e.g., website content summaries, screenshot analysis).
*   **Refactor the code** from a single notebook into a more modular Python project structure.
*   **Deploy the trained model** as an API for real-time inference.

## Contributions

Feel free to fork this repository

analysis by xanthorox 
