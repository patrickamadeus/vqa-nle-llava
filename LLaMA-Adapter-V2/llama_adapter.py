import llama
import torch
from PIL import Image
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "./LLaMA-7B/"

# Constants
LR_PROMPT_PATH = "../prompt/list-then-rewrite.txt"
QG_PROMPT_PATH = "../prompt/question-generation.txt"

with open(LR_PROMPT_PATH, "r") as file:
    LR_PROMPT= file.read()

with open(QG_PROMPT_PATH,"r") as file:
    QG_PROMPT = file.read()
    


# choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
model, preprocess = llama.load("LORA-BIAS-7B", llama_dir, llama_type="7B", device=device)
model.eval()

imageFile = "../dataset/003.jpg"
img = Image.open(imageFile)
img = preprocess(img).unsqueeze(0).to(device)
NUM = "10"

s = time.time()
lr_prompt = llama.format_prompt(LR_PROMPT.format(number = NUM))
lr_result = model.generate(img, [lr_prompt], 
                            max_gen_len=512, 
                            temperature=0.5, 
                            top_p=0.85)[0]
e = time.time()
lr_time = e-s
print(lr_result)


s = time.time()
qg_prompt = llama.format_prompt(QG_PROMPT.format(desc = lr_result, number = NUM))
qg_result = model.generate(img, [qg_prompt], 
                            max_gen_len=512, 
                            temperature=0.5, 
                            top_p=0.85)[0]
e = time.time()
qg_time = e-s
print()
print(qg_result)
             
             
             
# Result printing
LR_RESULT_FILENAME = f"LAV2_LR_result_{imageFile.split('/')[-1].split('.')[0]}.txt" 
QG_RESULT_FILENAME = f"LAV2_QG_result_{imageFile.split('/')[-1].split('.')[0]}.txt" 

with open(f"../result/LLaMa-Adapter-V2/{LR_RESULT_FILENAME}", "w") as file:
    # Writing data to a file
    file.write(lr_result)
    file.write("\n\n")
    file.write(f"Processing time : {lr_time}s")

with open(f"../result/LLaMa-Adapter-V2/{QG_RESULT_FILENAME}", "w") as file:
    # Writing data to a file
    file.write(qg_result)
    file.write("\n\n")
    file.write(f"Processing time : {qg_time}s")
