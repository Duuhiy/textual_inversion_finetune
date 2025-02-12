from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline

# pipe = StableDiffusionPipeline.from_pretrained(
# 	"CompVis/stable-diffusion-v1-4").to("cuda")
# prompt = "a photo of an astronaut riding a horse on mars"
# with autocast("cuda"):
#     image = pipe(prompt)["sample"][0]
#
# image.save("astronaut_rides_horse.png")



pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]