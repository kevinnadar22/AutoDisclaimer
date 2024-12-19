import moondream as md
from PIL import Image

# initialize with a downloaded model
model = md.vl(model="moondream-0_5b-int8.mf")

# process the image
image = Image.open("single.png")
encoded = model.encode_image(image)

# query the image
result = model.query(encoded, "is there any smoking scene")
print("Answer: ", result["answer"])
