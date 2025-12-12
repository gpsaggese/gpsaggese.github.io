# EconML: Analyzing the Effects of Education Programs on Student Performance

This project demonstrates how to use the **EconML** library to estimate causal effects, with a focus on a student education program and its impact on final grades.

## Files in This Folder

- `econml.API.ipynb` – Notebook demonstrating the EconML APIs and the custom wrapper in `econml_utils.py`.
- `econml.API.md` – Markdown documentation describing the API design and how to use it.
- `econml.example.ipynb` – End-to-end example analyzing the effect of an educational program on student performance.
- `econml.example.md` – Written explanation of the example, results, and evaluation.
- `econml_utils.py` – Utility module exposing a small wrapper API around EconML estimators for this project.
- `requirements.txt` – Python dependencies for this project.

## .API. vs .example. files

- **.API. files** focus on the **programmatic interface** (wrapper design + EconML usage).
- **.example. files** focus on the **applied case study** using the student performance dataset.