# Fine-tune Llama 3.2 Vision on Radiology Images

Fine-tunes the `unsloth/Llama-3.2-11B-Vision-Instruct` vision-language model to describe radiology images, and includes a small web app to try the result.

## Notebook (`Finetune_llama3_2vision.ipynb`)

1. **Load the model** in 4-bit with Unsloth's `FastVisionModel`.
2. **Add LoRA adapters** (rank 16, alpha 16) to the vision layers, language layers, attention modules and MLP modules.
3. **Load the data**: the `unsloth/Radiology_mini` dataset from Hugging Face. Each image and its caption becomes a conversation, with the instruction "You are an expert radiographer. Describe accurately what you see in this image."
4. **Train** with TRL's `SFTTrainer`: batch size 2 per device, 4 gradient accumulation steps, learning rate 2e-4, `adamw_8bit` optimiser, 30 steps, maximum sequence length 2048.
5. **Test** by generating a description for an image.
6. **Save** the LoRA adapter to `lora_model` and a merged model, and push the merged model to the Hugging Face Hub as `JVishu/Llama-3.2-11B-Vision-Radiology-mini`.

## Web app (`webui.py`)

A [Gradio](https://www.gradio.app) app that loads `JVishu/Llama-3.2-11B-Vision-Radiology-mini`. You upload an image, optionally edit the instruction, and it generates a description. The app is written with the older `gr.inputs` Gradio API.

```bash
pip install gradio transformers torch pillow requests
python webui.py
```

## Run the notebook

Open the notebook in Google Colab with a GPU runtime. The first cells install Unsloth and ask for a Hugging Face token. Add your own token to push the model.
