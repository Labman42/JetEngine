import os
from jetengine import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("<your path to model>/SDAR-4B-Chat")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    sdar_block_size = 4
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1, mask_token_id=151669, block_length=sdar_block_size, max_num_seqs=32, max_model_len=4096, gpu_memory_utilization=0.8) # Must set mask_token_id & block_length
    sampling_params = SamplingParams(temperature=1.0, topk=0, topp=1.0, max_tokens=4096,
                                     remasking_strategy="low_confidence_dynamic", dynamic_threshold=0.9,
                                     block_length=sdar_block_size, denoising_steps=sdar_block_size)

    questions = [
        "Consider the geometric sequence $\\frac{125}{9}, \\frac{25}{3}, 5, 3, \\ldots$. What is the eighth term of the sequence? Express your answer as a common fraction.",
        "A regular pentagon is rotated counterclockwise about its center. What is the minimum number of degrees it must be rotated until it coincides with its original position?",
        "If a snack-size tin of peaches has $40$ calories and is $2\\%$ of a person's daily caloric requirement, how many calories fulfill a person's daily caloric requirement?",
    ]

    prompts_list = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + "\n Solve the problem step by step.\n"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in questions
    ]
    
    outputs = llm.generate_streaming(
        prompts_list, sampling_params, max_active=128)

    for output in outputs:
        print("\n")
        print(f"Completion: {output['text']!r}")
        print(f"Total Length: {len(output['token_ids'])!r}")


if __name__ == "__main__":
    main()
