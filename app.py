import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from rembg import remove
from transformers import pipeline

# 1. Load local models onto your NVIDIA GPU ⚡
chat_model = pipeline(
    "text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"
)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to("cuda")


def process_workflow(user_message, history, input_image):
  if input_image is None:
    return (
        history
        + [(user_message, "Please upload a product image first using the box!")],
        None,
    )

  # 2. Format conversation history for Qwen 📜
  messages = [
      {
          "role": "system",
          "content": (
              "You are a helpful product photography assistant. Chat warmly with"
              " the user, but you MUST wrap the background image generation"
              " prompt inside <prompt> and </prompt> tags at the end of your"
              " reply."
          ),
      }
  ]
  for human, assistant in history:
    messages.append({"role": "user", "content": human})
    messages.append({"role": "assistant", "content": assistant})
  messages.append({"role": "user", "content": user_message})

  # 3. Generate chat response and isolate the image prompt 🤖
  response = chat_model(
      messages, max_new_tokens=256, do_sample=True, temperature=0.7
  )
  full_output = response[0]["generated_text"][-1]["content"]

  if "<prompt>" in full_output and "</prompt>" in full_output:
    chat_reply = full_output.split("<prompt>")[0].strip()
    image_prompt = (
        full_output.split("<prompt>")[1].split("</prompt>")[0].strip()
    )
  else:
    chat_reply = full_output
    image_prompt = "professional studio background, clean lighting, photorealistic"

  # 4. Remove background and build the inpainting mask ✂️
  target_size = (512, 512)
  resized_input = input_image.resize(target_size)
  output_cutout = remove(resized_input)

  # Invert alpha to create mask: Product = Black (protected), Background = White (editable) 🗺️
  alpha = output_cutout.split()[3]
  silhouette = alpha.point(lambda p: 255 if p > 0 else 0).convert("L")
  mask_image = Image.eval(silhouette, lambda px: 255 - px).convert("RGB")

  # 5. Run local inpainting 🎨
  result_image = pipe(
      prompt=image_prompt,
      image=output_cutout.convert("RGB"),
      mask_image=mask_image,
      height=512,
      width=512,
  ).images[0]

  updated_history = history + [(user_message, chat_reply)]
  return updated_history, result_image


# 6. Build the Gradio UI Layout 🖥️
with gr.Blocks() as demo:
  gr.Markdown("# 📸 AI Product Photography Assistant")

  with gr.Row():
    with gr.Column(scale=1):
      image_input = gr.Image(
          type="pil", label="Upload Product Image", interactive=True
      )
      image_output = gr.Image(type="pil", label="Generated Background")

    with gr.Column(scale=1):
      chatbot = gr.Chatbot(label="Chat Assistant")
      msg_input = gr.Textbox(
          label="Type instructions for your product background..."
      )
      submit_btn = gr.Button("Send")

      submit_btn.click(
          fn=process_workflow,
          inputs=[msg_input, chatbot, image_input],
          outputs=[chatbot, image_output],
      )
      msg_input.submit(
          fn=process_workflow,
          inputs=[msg_input, chatbot, image_input],
          outputs=[chatbot, image_output],
      )

if __name__ == "__main__":
  demo.launch()
