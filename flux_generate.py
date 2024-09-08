import torch
from diffusers import FluxPipeline
import diffusers

# Modify the rope function to handle MPS device (macs)
_flux_rope = diffusers.models.transformers.transformer_flux.rope
def new_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    if pos.device.type == "mps":
        return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
    else:
        return _flux_rope(pos, dim, theta)

diffusers.models.transformers.transformer_flux.rope = new_flux_rope

# Load the Flux model -- note the use of both bfloat16 and mps. I can't say why
# but these seem to be required for the model to run on my mac.
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("mps")

# Set the prompt for image generation. don't need the negative prompt but I find
# it to be more effective to include it.
prompt = "An English gentleman in bat form. It should have wings and a cape and be wearing a top hat. He should be drinking a cup of tea. negative prompt: not a bat, not a man, not a gentleman, no wings, no cape, no top hat, no tea, no cup of tea"

# Generate the image
out = pipe(
    # Our prompt above
    prompt=prompt,
    # This can be seen as a measure of creativity. 0 is very creative, 1 is not
    # creative at all
    guidance_scale=0,
    # The dimensions of the image output. This needs to be a multiple of 8.
    # Smaller image means faster inference.
    height=400,
    width=400,
    # Number of steps the model will take to generate the image. More steps
    # means less noise and more detail but can also result in slower inference.
    num_inference_steps=12,
    # This is the maximum length of the sequence. Flux is a sequence prediction
    # model so it needs to know the maximum length of the sequence it will be
    # dealing with. Sequence length is the maximum length of the prompt + the
    # maximum length of the output. Higher value here means the model can deal
    # with longer sequences thus make better images but will take longer.
    max_sequence_length=256,
).images[0]

# Save the generated image
out.save("flux_image.png")

# Display the image
out.show()