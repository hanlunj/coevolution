# Coevolution

This repository contains the codebase for the manuscript **"[Structural ontogeny of protein-protein interactions]()"**.  
It provides tools for:

- Training a **Selection Probability Model (SPM)**
- Computing **epistasis** from predicted fitness landscapes
- Simulating **coevolutionary trajectories**

All code and model parameters are distributed under the **MIT License**.

---

### Installation

```bash
conda create -n coevolution python=3.9.18
conda activate coevolution
pip install -r requirements.txt
```

---

### Training and Using SPM

Code for training and applying the SPM is available in the `spm/` directory.  
Training data and pre-trained model weights can be downloaded from the [Zenodo repository]() and placed under `spm/data/` and `spm/models/`.

An example of running the training script and using the trained model for prediction can be found in `spm/example/`.

---

### Computing Epistasis

Code for computing epistasis from an SPM-predicted fitness landscape is located in `epistasis/`.  
A predefined sequence space and precomputed fitness landscape are available at `epistasis/data/` via the [Zenodo repository]().

The main output file, **`term_extractor_fitness_all.pickle`**, contains computed marginals and effect sizes (referred to as *factors* in the code) stored in a Python dictionary.

---

### Simulating Coevolutionary Trajectories

Code for simulating coevolutionary trajectories and computing energy barriers between variants is available in `simulation/`.  
Input files—including the predefined sequence space, one-hot encoded sequences, and precomputed fitness values—can be found at `simulation/data/` in the [Zenodo repository]().

The output files **`sink_freqs.npz`** and **`sinks.dict.pkl`** contain sequences with nonzero accessibilities (termed *sinks* in the code) along with their accessibility values.

---

### Citation

If you use this code or data in your research, please cite:
**"[Structural ontogeny of protein-protein interactions]()"**  
> [Full citation to be added once the manuscript is published.]

---

### License

This project is licensed under the [MIT License](LICENSE).

