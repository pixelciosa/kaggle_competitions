# Personality Predictor
This project is a comparisson of different binary classification models, trained with different datasets
using the data of [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e7/) 

The best result was **XGBoost using Bagging** and no transformations are needed on the data, so it can include datapoints with NaN and unscaled values.

### Accuracy:
Score: 0.975708

Private score: 0.968421

## Environment
- Python 3.11.13
- Virtual environment: `.venv`
- Complete dependencies listed on `requirements.txt` (for exact installation)
- Main dependencies listed on `requirements_main.txt` (for documentation and reading)

## Recreate environment

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run app locally
- Execute from project's root folder: `python app.py`
- To test the API locally, type in the project folder in the Terminal: `python call-api.py`


## Open app in HuggingFace:
- https://huggingface.co/spaces/pixelciosa/Personality_prediction

## Example to test API
```python
client = Client("http://127.0.0.1:7860/")
result = client.predict(
	param_0=1,
	param_1=False,
	param_2=5,
	param_3=6,
	param_4=True,
	param_5=4,
    param_6=2,
	api_name="/predict"
)
print(result)
```

