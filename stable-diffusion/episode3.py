# https://huggingface.co/Justin-Choo/epiCRealism-Natural_Sin_RC1_VAE

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Justin-Choo/epiCRealism-Natural_Sin_RC1_VAE")

pipe = pipe.to("mps")

pipe.enable_attention_slicing()
prompt = "polaroid photo, night photo, photo of 24 y.o beautiful woman, pale skin, bokeh, motion blur"

pipe(prompt, num_inference_steps=15)

image = pipe(prompt).images[0]

image.save("./episode3.jpg")


