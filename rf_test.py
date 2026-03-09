from inference_sdk import InferenceHTTPClient
import os
import sys

MODEL_ID = "individual-classification/1"

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

# get image path from terminal
img_path = sys.argv[1]

result = client.infer(img_path, model_id=MODEL_ID)

pred_class = result.get("top")
confidence = result.get("confidence")

print("\nPrediction")
print("----------------")
print(f"Class: {pred_class}")
print(f"Confidence: {confidence:.3f}")