from transformers import AutoTokenizer
from .modeling_minicpm import MiniCPM3ForCausalLM
import mindspore as ms
ms.set_seed(0)

path = 'openbmb/MiniCPM3-4B'
tokenizer = AutoTokenizer.from_pretrained(path)
model = MiniCPM3ForCausalLM.from_pretrained(path, mindspore_dtype=ms.bfloat16)

responds, history = model.chat(tokenizer, "请写一篇关于人工智能的文章，详细介绍人工智能的未来发展和隐患。", temperature=0.7, top_p=0.7)
print(responds)