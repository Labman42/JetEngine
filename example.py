import os
from jetengine import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    """
    Example usage of JetEngine with SDAR model for mathematical problem solving.
    
    This example demonstrates:
    - Loading a pre-trained SDAR model
    - Configuring sampling parameters for block diffusion
    - Running inference on mathematical problems
    - Using streaming generation for batch processing
    """
    
    # Configuration - adjust these paths as needed for your setup
    model_path = os.path.expanduser("~/models/SDAR-1.7B-Chat")  # Update this path
    
    try:
        # Initialize tokenizer and LLM
        print(f"Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(
            model_path, 
            enforce_eager=False, 
            tensor_parallel_size=1, 
            mask_token_id=151669,  # Required for SDAR models
            block_length=16
        )
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model exists at: {model_path}")
        print("You can download the model using:")
        print("huggingface-cli download --resume-download JetLM/SDAR-1.7B-Chat --local-dir ~/models/SDAR-1.7B-Chat/")
        return
    
    # Configure sampling parameters for block diffusion
    sampling_params = SamplingParams(
        temperature=1.0, 
        topk=0, 
        topp=1.0, 
        max_tokens=4096,
        remasking_strategy="low_confidence_dynamic", 
        block_length=16, 
        denoising_steps=16, 
        dynamic_threshold=0.9
    )

    # Example prompts - mathematical problems for demonstration
    prompts = [
        "A math club is having a bake sale as a fundraiser to raise money for an upcoming trip. "
        "They sell 54 cookies at three for $1, and 20 cupcakes at $2 each, and 35 brownies at $1 each. "
        "If it cost the math club $15 to bake these items, what was their profit?\n"
        "Please reason step by step, and put your final answer within \\boxed{}.",
        
        "A 90° rotation around the origin in the counter-clockwise direction is applied to 7 + 2i. "
        "What is the resulting complex number?\n"
        "Please reason step by step, and put your final answer within \\boxed{}.",
        
        "Simplify √242.\n"
        "Please reason step by step, and put your final answer within \\boxed{}.",
        
        "Consider the geometric sequence 125/9, 25/3, 5, 3, .... "
        "What is the eighth term of the sequence? Express your answer as a common fraction.\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    ]
    
    try:
        # Apply chat template to prompts
        print("Preparing prompts...")
        prompts_list = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True
            )
            for prompt in prompts
        ]
        
        # Generate responses using streaming mode
        print("Starting inference...")
        outputs = llm.generate_streaming(prompts_list, sampling_params, max_active=64)
        
        # Display results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\nProblem {i+1}:")
            print("-" * 40)
            print(f"Question: {prompt}")
            print(f"\nAnswer: {output['text']}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return
        
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
