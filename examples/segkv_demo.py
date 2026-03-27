import time
from vllm import LLM, SamplingParams

def main():
    print("Initializing vLLM with SegKV...")
    llm = LLM(
        model="facebook/opt-125m",
        enable_segkv=True,
        segkv_segment_size=64,
        segkv_blend_max_layer_frac=0.5,
    )
    
    doc_id = "demo_document"
    base_text = "The quick brown fox jumps over the lazy dog. " * 50
    
    print("\n--- Registering Base Document ---")
    reg_info = llm.register_document(doc_id, base_text)
    print(f"Registered {doc_id} version {reg_info['version']} "
          f"({reg_info['num_segments']} segments, {reg_info['total_tokens']} tokens)")
          
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    
    print("\n--- First Generation (No Cache Hit Expected) ---")
    start = time.perf_counter()
    outputs = llm.generate([base_text + "\nQ: What jumps over the lazy dog?\nA:"], sampling_params)
    end = time.perf_counter()
    print(f"Output: {outputs[0].outputs[0].text.strip()}")
    print(f"Time: {end - start:.3f}s")
    
    print("\n--- Editing Document ---")
    new_text = "The completely fast brown cat jumps over the lazy dog. " * 50
    
    print("\n--- Planning Edit (CPU only) ---")
    plan_info = llm.plan_edit(doc_id, new_text)
    print(f"Edit will bump version {plan_info['old_version']} to {plan_info['new_version']}")
    print(f"Strategy Summary: {plan_info['summary']}")
    print(f"Estimated Savings: {plan_info['estimated_savings'] * 100:.1f}% vs full recompute")
    
    print("\n--- Second Generation (Cache Hit & CacheBlend Expected) ---")
    start = time.perf_counter()
    outputs2 = llm.generate([new_text + "\nQ: What jumps over the lazy dog?\nA:"], sampling_params)
    end = time.perf_counter()
    print(f"Output: {outputs2[0].outputs[0].text.strip()}")
    print(f"Time: {end - start:.3f}s")
    
    print("\n--- Third Generation (Pure Prefix Match Expected) ---")
    start = time.perf_counter()
    outputs3 = llm.generate([new_text + "\nQ: Who is lazy?\nA:"], sampling_params)
    end = time.perf_counter()
    print(f"Output: {outputs3[0].outputs[0].text.strip()}")
    print(f"Time: {end - start:.3f}s")

if __name__ == "__main__":
    main()
