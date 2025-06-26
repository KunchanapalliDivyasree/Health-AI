import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ IBM Granite model setup
model_id = "ibm-granite/granite-3.3-2b-instruct"
token = os.getenv("HF_TOKEN")  # Ensure your Hugging Face token is set in the environment

# ✅ Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    device_map="auto",
    torch_dtype=torch.float32
)
# ✅ Query function
def query_granite(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Gradio UI using Tabs (instead of Pages)
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Welcome to HealthAI")
    gr.Markdown("Your intelligent healthcare assistant.")

    with gr.Tab("🩺 Symptoms"):
        def identify(symptom):
            return query_granite(f"What illness could cause: {symptom}?")
        symptom = gr.Textbox(label="Enter your symptom")
        output = gr.Textbox(label="AI Diagnosis")
        btn = gr.Button("Analyze")
        btn.click(identify, inputs=symptom, outputs=output)

    with gr.Tab("🌿 Remedies"):
        def get_remedies(issue):
            return query_granite(f"What are home remedies for {issue}?")
        issue = gr.Textbox(label="What are you suffering from?")
        remedy_output = gr.Textbox(label="Suggested Remedy")
        remedy_btn = gr.Button("Suggest")
        remedy_btn.click(get_remedies, inputs=issue, outputs=remedy_output)

    with gr.Tab("🥗 Diet"):
        def suggest(goal):
            return query_granite(f"Suggest a diet for: {goal}")
        goal = gr.Textbox(label="Your health goal")
        diet_output = gr.Textbox(label="Diet Plan")
        diet_btn = gr.Button("Get Plan")
        diet_btn.click(suggest, inputs=goal, outputs=diet_output)

    with gr.Tab("🧠 Mental Wellness"):
        def tip(topic):
            return query_granite(f"Mental health advice about: {topic}")
        topic = gr.Textbox(label="Enter mental health topic")
        tip_output = gr.Textbox(label="Wellness Tip")
        tip_btn = gr.Button("Get Tip")
        tip_btn.click(tip, inputs=topic, outputs=tip_output)

    with gr.Tab("❓ FAQs"):
        gr.Markdown("### ❓ FAQs")
        gr.Markdown("**Q1:** What is HealthAI?")
        gr.Markdown("**A:** It's an AI assistant to help with health-related queries using IBM Granite 3.3-2B.")

demo.launch()

