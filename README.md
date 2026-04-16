# Project Setup

## Initial Setup

1. **Build and Launch Containers**:
   - Run the `build` and `launch_container` scripts located in the `container_setups` folder to build and start the necessary containers.
   
2. **Environment Variables**:
   - Create a `.env` file in the project root directory.
   - Add the following variables to the `.env` file:
     ```
     HUGGING_FACE_API_KEY=your_hugging_face_api_key_here
     COMET_API_KEY=your_comet_api_key_here
     ```
## Directory Structure

- **config/**:
  - Contains all `.yaml` configuration files for different steps in the pipeline.
  - The pipeline consists of two main parts:
    1. **preprocess**: Configuration files for dataset preprocessing scripts.
    2. **method**: Configuration files for both supervised and unsupervised methods used in hallucination detection.

- **container_setups/**:
  - Contains scripts and files needed for building and launching containers.

- **data/**:
  - Stores raw and preprocessed datasets.

- **cache/**:
  - Caches computed data, including hidden states, to prevent redundant computations.

- **src/**:
  - Contains all the source code for the pipeline, including preprocessing, model training, and evaluation scripts.

## Running the Pipeline

- To run the pipeline, execute the run script:
  ```bash
  python run_mtopdiv.py
  ```
