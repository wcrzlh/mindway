import mindspore as ms
import numpy as np

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from mindway.transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

if __name__ == "__main__":
    # Debug and testing use only

    import time

    # ms.set_context(mode=ms.PYNATIVE_MODE)
    # ms.runtime.launch_blocking()
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT) # NOT SUPPORTED YET
    print(f"mode: {ms.get_context('mode')}, device: {ms.get_context('device_target')}")
    # TEST: loading model
    start_time = time.time()
    config = Qwen3Config(
        num_hidden_layers=1,
        use_cache=False,
        attn_implementation="flash_attention_2",
        sliding_window=None,
    )

    model = Qwen3ForCausalLM(config)
    print("*" * 100)
    print("Test passed: Sucessfully loaded Qwen3ForCausalLM")
    print("Time elapsed: %.4fs" % (time.time() - start_time))
    print("*" * 100)

    # TEST: process input
    input_ids = ms.Tensor([[10, 15, 41]], ms.int32)
    # prompt = ["are you conscious?",]
    # input_ids = ms.Tensor(tokenizer(prompt).input_ids, ms.int32)

    input_kwargs = {}
    input_kwargs["input_ids"] = input_ids

    output_ids = model.generate(**input_kwargs, max_new_tokens=5, do_sample=False)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, output_ids)
    ]

    print(f"generated id: {generated_ids}")
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


