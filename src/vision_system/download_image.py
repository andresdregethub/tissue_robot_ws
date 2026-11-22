
from datasets import load_dataset

# Download the face dataset from Hugging Face
dataset = load_dataset(
    "DeviL1337/face_recognition_52",
    split="train"
)

# Get the first image
image = dataset[0]["image"]

# Save the image locally
image.save("images/test.jpg")

print("Face image downloaded successfully!")
print("Saved to images/test.jpg")

