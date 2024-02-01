from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
# result = client.predict(
# 		"C:\\Users\\hp\\Downloads\\im1.jpeg",api_name="/predict"	
# )

result = client.predict({"video":"C:\\Users\\hp\\Downloads\\ras_df.mp4",
                        "subtitles":None},	api_name="/predict_1")
print(result)