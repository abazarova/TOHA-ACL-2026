# 🚀 Hallucination Detection in LLMs with Topological Divergence on Attention Graphs

🎉 **Exciting News: This paper has been officially accepted to ACL 2026!** 🎉

Welcome to the official code repository for our paper! This repository contains the complete implementation of our proposed method for hallucination detection, including dataset preprocessing, MTop-Div feature extraction, and the TOHA evaluation pipeline.

🔗 **[Read the Paper on arXiv (Link)](https://arxiv.org/abs/2504.10063)** 

---

## 🛠️ Project Setup

### 1. Initial Setup

*   **🐳 Build and Launch Containers**: 
    To ensure a consistent environment, we use containers. Run the `build` and `launch_container` scripts located in the `container_setups` directory to build and start the necessary containers.
 
*   **🔐 Environment Variables**:
    Create a `.env` file in the project root directory to store your credentials securely. Add the following variables to your `.env` file:
    ```env
    HUGGING_FACE_API_KEY=your_hugging_face_api_key_here
    COMET_API_KEY=your_comet_api_key_here
    ```

### 2. Data Preparation

*   **📄 Load Datasets**: 
    Prepare your raw data files in `.csv` format and place them into your working data directory.
*   **⚙️ Configure Data**: 
    Once your `.csv` files are ready, create or update the corresponding configuration files inside the `config/data/` folder and preprocessing files inside the `src/preprocess` folder so the pipeline knows how to load and parse your specific datasets.

---

## 📁 Directory Structure

Here is a quick overview of how the repository is organized:

*   **`config/`** ⚙️
    *   Contains all `.yaml` configuration files for the different steps in the pipeline.
    *   The configuration is split into main parts:
        1.  **`data/`**: Configurations mapped to your raw `.csv` data files.
        2.  **`preprocess/`**: Settings for dataset downloading and preprocessing scripts.
        3.  **`method/`**: Settings for running both supervised baselines and our unsupervised methods for hallucination detection.
*   **`container_setups/`** 📦
    *   Contains scripts and Dockerfiles needed for building and launching the reproducible container environment.
*   **`src/`** 💻
    *   Contains all the core source code, including preprocessing, model inference, topological feature computation, and evaluation scripts.

---

## ▶️ Running the Pipeline

Once your container is running, your `.env` variables are set, and your `.csv` data configs are ready, you can easily execute the main pipeline. 

To run the experiments, simply execute:
```bash
python run_mtopdiv.py
```

---

## 📜 Citation

```bibtex
@article{bazarova2025hallucination,
  title={Hallucination detection in llms with topological divergence on attention graphs},
  author={Bazarova, Alexandra and Yugay, Aleksandr and Shulga, Andrey and Ermilova, Alina and Volodichev, Andrei and Polev, Konstantin and Belikova, Julia and Parchiev, Rauf and Simakov, Dmitry and Savchenko, Maxim and others},
  journal={arXiv preprint arXiv:2504.10063},
  year={2025}
}
```
