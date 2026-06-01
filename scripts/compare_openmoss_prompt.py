#!/usr/bin/env python3
"""Compare openmoss C++ prompt vs Python MOSS chat template."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("OpenMOSS-Team/MOSS-TTS-v1.5", trust_remote_code=True)

text = "Hello world test one two three."
body = f"""<user_inst>
- Reference(s):
None
- Instruction:
None
- Tokens:
None
- Quality:
None
- Sound Event:
None
- Ambient Sound:
None
- Language:
English
- Text:
{text}
</user_inst>"""

im_start = tok.decode([151644])
im_end = tok.decode([151645])
audio_start = tok.decode([151652])

openmoss_prompt = im_start + "user\n" + body + im_end + "\n" + im_start + "assistant\n" + audio_start

user_msg = {"role": "user", "content": body}
py_prompt = tok.apply_chat_template([user_msg], add_generation_prompt=True, tokenize=False)

print("OPENMOSS chars:", len(openmoss_prompt))
print("PYTHON chars:", len(py_prompt))
print("EQUAL:", openmoss_prompt == py_prompt)
print("--- openmoss ---")
print(repr(openmoss_prompt))
print("--- python ---")
print(repr(py_prompt))

ids_o = tok.encode(openmoss_prompt, add_special_tokens=False)
ids_p = tok.encode(py_prompt, add_special_tokens=False)
print("token counts openmoss/python:", len(ids_o), len(ids_p))
if ids_o != ids_p:
    n = min(len(ids_o), len(ids_p))
    for i in range(n):
        if ids_o[i] != ids_p[i]:
            print("first diff at", i, ids_o[i], ids_p[i], repr(tok.decode([ids_o[i]])), repr(tok.decode([ids_p[i]])))
            break
    print("openmoss tail:", ids_o[-5:], [tok.decode([x]) for x in ids_o[-5:]])
    print("python tail:", ids_p[-5:], [tok.decode([x]) for x in ids_p[-5:]])
