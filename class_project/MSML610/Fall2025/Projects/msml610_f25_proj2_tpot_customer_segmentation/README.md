# MSML610 Project 2 – TPOT-based Customer Segmentation

## Objective
Use the UCI Online Retail dataset to build an end-to-end machine learning pipeline that:
- Cleans transactional data
- Aggregates customers into behavioral features
- Uses TPOT (AutoML) to learn an optimal preprocessing pipeline
- Applies clustering to derive interpretable customer segments

## Dataset
Online Retail Data Set (UCI Machine Learning Repository):

Users must download `Online Retail.xlsx` separately and place it in the `data/` folder.

## How to Run

```bash
pip install -r requirements.txt
python -m src.main
