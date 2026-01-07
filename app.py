import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gc
import os
from huggingface_hub import snapshot_download
from pathlib import Path

# Cache directory for models and adapters
CACHE_DIR = Path.home() / ".cache" / "lora_assemble"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# LoRA Adapter Configuration
ADAPTERS = {
    "predibase/cnn": {
        "name": "CNN News Summarization",
        "description": "Generates summaries in CNN/DailyMail style",
        "example": "Summarize: The researchers discovered a new species of deep-sea fish..."
    },
    "predibase/conllpp": {
        "name": "Named Entity Recognition",
        "description": "Identifies and extracts named entities from text",
        "example": "Extract entities: Apple CEO Tim Cook announced the new iPhone in Cupertino."
    },
    "predibase/magicoder": {
        "name": "Code Generation",
        "description": "Generates code solutions for programming tasks",
        "example": "Write a Python function to find the factorial of a number."
    },
    "predibase/agnews_explained": {
        "name": "News Classification",
        "description": "Classifies news articles with explanations",
        "example": "Classify: Tech giants report record profits in Q4 earnings."
    },
    "predibase/customer_support": {
        "name": "Customer Support",
        "description": "Generates helpful customer support responses",
        "example": "Customer: My order hasn't arrived yet. It's been 2 weeks."
    },
    "predibase/tldr_headline_gen": {
        "name": "Headline Generation",
        "description": "Creates catchy headlines from article content",
        "example": "Generate headline: Scientists have discovered water on Mars..."
    },
    "predibase/tldr_content_gen": {
        "name": "TL;DR Summarization",
        "description": "Creates concise TL;DR summaries",
        "example": "TL;DR: The meeting covered quarterly results, new product launches..."
    },
    "predibase/wikisql": {
        "name": "SQL Query Generation",
        "description": "Converts natural language to SQL queries",
        "example": "Convert to SQL: Show all customers from New York with orders over $100"
    },
    "predibase/hellaswag": {
        "name": "Commonsense Reasoning",
        "description": "Completes scenarios with commonsense understanding",
        "example": "Complete: A person is cooking pasta. They boil water, add salt, then..."
    },
    "predibase/jigsaw": {
        "name": "Toxicity Detection",
        "description": "Detects and analyzes toxic content",
        "example": "Analyze toxicity: This comment contains harsh criticism of the author."
    },
    "predibase/drop_explained": {
        "name": "Reading Comprehension",
        "description": "Answers questions about passages with explanations",
        "example": "Passage: The Eiffel Tower is 330m tall... Question: How tall is it?"
    }
}

BASE_MODEL = "mistralai/Mistral-7B-v0.1"

# Global variables for model caching
model = None
tokenizer = None
current_adapter = None
adapter_cache = {}  # Cache for storing loaded adapter references


def preload_models_and_adapters():
    """
    Pre-download the base model and all LoRA adapters at startup.
    This ensures no downloads happen during user queries.
    """
    print("=" * 60)
    print("🚀 Pre-caching models and adapters...")
    print("=" * 60)
    
    # Pre-download base model
    print(f"\n📥 Downloading base model: {BASE_MODEL}")
    try:
        snapshot_download(
            repo_id=BASE_MODEL,
            cache_dir=str(CACHE_DIR / "models"),
            resume_download=True
        )
        print(f"✅ Base model cached successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-cache base model: {e}")
    
    # Pre-download all adapters
    for adapter_id, info in ADAPTERS.items():
        print(f"\n📥 Downloading adapter: {info['name']} ({adapter_id})")
        try:
            snapshot_download(
                repo_id=adapter_id,
                cache_dir=str(CACHE_DIR / "adapters"),
                resume_download=True
            )
            print(f"✅ Adapter '{info['name']}' cached successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not pre-cache adapter {adapter_id}: {e}")
    
    print("\n" + "=" * 60)
    print("✨ Pre-caching complete! App is ready for queries.")
    print("=" * 60 + "\n")


def load_base_model():
    """Load the base model with 4-bit quantization."""
    global model, tokenizer
    
    if model is not None:
        return model, tokenizer
    
    print("Loading base model with 4-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("Base model loaded successfully!")
    return model, tokenizer


def load_adapter(adapter_id: str):
    """Load a LoRA adapter onto the base model."""
    global model, current_adapter
    
    if current_adapter == adapter_id:
        return model
    
    base_model, _ = load_base_model()
    
    # Unload previous adapter if exists
    if current_adapter is not None:
        try:
            model = base_model.unload()
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Loading adapter: {adapter_id}...")
    model = PeftModel.from_pretrained(base_model, adapter_id)
    current_adapter = adapter_id
    print(f"Adapter {adapter_id} loaded successfully!")
    
    return model


def generate_response(
    prompt: str,
    adapter_id: str,
    temperature: float,
    max_tokens: int,
    progress=gr.Progress()
):
    """Generate a response using the selected adapter."""
    
    if not prompt.strip():
        return "Please enter a prompt."
    
    progress(0, desc="Loading model...")
    
    try:
        model = load_adapter(adapter_id)
        _, tokenizer = load_base_model()
        
        progress(0.3, desc="Generating response...")
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        progress(0.9, desc="Decoding...")
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        progress(1.0, desc="Done!")
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"


def update_info(adapter_id: str):
    """Update the adapter info display."""
    info = ADAPTERS.get(adapter_id, {})
    return info.get("description", ""), info.get("example", "")


# Custom CSS for styling
CUSTOM_CSS = """
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 1rem;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    color: white;
    margin: 0;
    font-size: 2.5rem;
}
.main-header p {
    color: rgba(255,255,255,0.9);
    margin-top: 0.5rem;
}
.adapter-card {
    background: linear-gradient(145deg, #f0f4ff, #e8eeff);
    border-radius: 0.75rem;
    padding: 1rem;
    border-left: 4px solid #667eea;
}
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-size: 1.1rem !important;
}
"""

# Custom theme
CUSTOM_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate"
)

# Build Gradio Interface
with gr.Blocks(title="LoRA Assemble") as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🔧 LoRA Assemble</h1>
            <p>Select a specialized LoRA adapter for Mistral-7B and generate responses</p>
        </div>
    """)
    
    with gr.Row():
        # Left Column - Controls
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Configuration")
            
            adapter_dropdown = gr.Dropdown(
                choices=[(ADAPTERS[k]["name"], k) for k in ADAPTERS.keys()],
                value="predibase/magicoder",
                label="Select LoRA Adapter",
                info="Choose a specialized adapter for your task"
            )
            
            with gr.Group():
                gr.Markdown("**Adapter Info**", elem_classes=["adapter-card"])
                adapter_desc = gr.Textbox(
                    label="Description",
                    value=ADAPTERS["predibase/magicoder"]["description"],
                    interactive=False,
                    lines=2
                )
                example_text = gr.Textbox(
                    label="Example Prompt",
                    value=ADAPTERS["predibase/magicoder"]["example"],
                    interactive=False,
                    lines=2
                )
            
            gr.Markdown("### ⚙️ Generation Settings")
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Higher = more creative, Lower = more focused"
            )
            
            max_tokens = gr.Slider(
                minimum=50,
                maximum=500,
                value=200,
                step=25,
                label="Max Tokens",
                info="Maximum length of generated response"
            )
        
        # Right Column - Input/Output
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Input & Output")
            
            prompt_input = gr.Textbox(
                label="Your Prompt",
                placeholder="Enter your prompt here...",
                lines=5
            )
            
            generate_btn = gr.Button(
                "🚀 Generate Response",
                variant="primary",
                size="lg",
                elem_classes=["generate-btn"]
            )
            
            output = gr.Textbox(
                label="Generated Response",
                lines=12
            )
            
            gr.Markdown("""
                > ⏱️ **Note**: Generation may take 30-60+ seconds on CPU.
            """)
    
    # Footer
    gr.Markdown("""
    ---
    **Base Model**: `mistralai/Mistral-7B-v0.1` | **Adapters by**: [Predibase](https://huggingface.co/predibase)
    """)
    
    # Event Handlers
    adapter_dropdown.change(
        fn=update_info,
        inputs=[adapter_dropdown],
        outputs=[adapter_desc, example_text]
    )
    
    generate_btn.click(
        fn=generate_response,
        inputs=[prompt_input, adapter_dropdown, temperature, max_tokens],
        outputs=[output]
    )


if __name__ == "__main__":
    # Pre-download all models and adapters at startup
    preload_models_and_adapters()
    demo.launch(theme=CUSTOM_THEME, css=CUSTOM_CSS)
