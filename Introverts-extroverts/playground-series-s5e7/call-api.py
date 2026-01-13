from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
# client = Client("pixelciosa/personality_prediction")
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