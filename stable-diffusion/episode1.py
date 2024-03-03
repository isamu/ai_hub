from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# for M1 mac
pipe = pipe.to("mps")

pipe.enable_attention_slicing()

prompt = "airplane and clouds"
n_prompt = "bad fingers"

# warmup for mac
pipe(prompt, negative_prompt=n_prompt,  num_inference_steps=1)

image = pipe(prompt, negative_prompt=n_prompt).images[0]

image.save("./episode1.jpg")

