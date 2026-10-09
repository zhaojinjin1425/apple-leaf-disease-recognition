Apple Leaf Disease Classification Using Enhanced Visual Features and Knowledge Graph
This repository contains the implementation of the method proposed in the paper:
Apple Leaf Disease Classification Using Enhanced Visual Features and Knowledge Graph
The proposed framework integrates enhanced visual feature representation with domain knowledge from a knowledge graph for apple leaf disease classification. The visual branch uses an improved EfficientNet-B0 to enhance disease-related visual representations, while the knowledge branch incorporates disease-related semantic knowledge. A query-key-based knowledge selection mechanism and bilinear interaction are employed to facilitate visual-knowledge fusion.

 1. Method Overview
The framework consists of three main components:
Enhanced visual feature extraction: An improved EfficientNet-B0 is used to extract discriminative visual features from apple leaf images.
Knowledge graph representation: Disease-related knowledge is organized into triples and embedded into a continuous feature space.
Visual-knowledge fusion: A query-key-based knowledge selection mechanism is used to select relevant knowledge according to the input image, followed by bilinear interaction for visual-knowledge fusion.

The overall framework is designed for five apple leaf disease categories.

2. Environment
The implementation was developed and tested with:
Python 3.x
PyTorch 2.0.0
torchvision 0.15.1
CUDA 11.8
Install the required Python packages using:
pip install -r requirements.txt

4. Dataset
The experiments use publicly available apple leaf disease datasets.
The dataset contains six classes, including five disease categories and healthy leaves. The knowledge graph is constructed for the five disease categories.
The datasets used in this study include:
PlantVillage
AppleLeaf9
Plant Pathology 2021-FGVC8
Please obtain the datasets from their original sources and organize them according to the paths required by the training scripts.
