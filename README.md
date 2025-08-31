# Breath Pattern Modeling and Pattern Recognition for Health Anomaly Detection 

This project simulates, distorts, and classifies respiratory flow patterns to enable health anomaly detection based on exhaled breath dynamics. It is part of the module "Research Project Advanced Network Technologies" at TU Berlin (SS2025). The system models synthetic breath signals, simulates signal distortion through a molecular channel, and classifies the resulting patterns using machine learning. It includes modules for:

- Synthetic signal generation (e.g., eupnea, apnea, tachypnea, Cheyne-Stokes, etc.)
- Signal distortion (e.g., distance-based attenuation, noise models)
- Dataset generation (samples and sequences)
- Classification using Random Forest or Neural Networks

The framework is written in Python and designed to be modular and extensible.

## Installation

1. Clone the repository:

```
git clone https://git.tu-berlin.de/tkn/teaching/pj-2025s/06-breath-pattern-modeling-and-pattern-recognition-health-anomaly-detection.git
```
2. Create a virtual environment (optional but recommended). We tested our code on Python 3.13.0 on MacOS 15.5:
```
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3.	Install dependencies:
```
pip install -r requirements.txt
```

4. In order to use our code run the following line
```
pip install -e .
```

## Usage
Create a dataset by running
```
python src/breath_modeling/model/create_dataset.py
```
You can customize parameters such as sampling rate, breathing types, window size, and noise in `create_dataset.py.

We also created a `main.py` script and a companion notebook `main.ipynb`. The notebook provides detailed explanations of the core functionalities, including usage examples and visualizations. The main.py script demonstrates a minimal working example of the system, but with significantly less explanation. For a deeper understanding of the theoretical background, we recommend reading the project report.

To run the main, run the following code:
```
python main.py
```

## Contributing
Contributions are welcome! Please fork the repository and open a merge request.

## Authors and acknowledgment
Developed by Ramin Leon Neymeyer, Anton Dilg and Ramon Rennert
Supervised by: Dr.-Ing Sunasheer Bhattacharjee

## License
This project is for educational use within the TU Berlin course “Research Project Advanced Network Technologies”. No commercial use allowed.

## Project status
The core project is completed, but there are a few implementations that would be valuable additions moving forward.
1. Retrain or redesign the architecture to support an attention-based model
2. Develop a graphical user interface (GUI) for easier interaction and usability
3. Integrate explainable AI techniques to evaluate model decisions and improve interpretability
4. Retrain and validate the model on real measured (sensor) data
5. Implement denoising methods to improve signal robustness and model accuracy