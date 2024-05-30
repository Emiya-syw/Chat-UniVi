import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math
import glob
import clip
import numpy as np

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_sorted_files(directory, pattern="*9.MP4*"):
    """
    使用 glob 搜索包含特定模式的文件，并按文件名排序
    :param directory: 要搜索的目录
    :param pattern: 文件名中包含的模式
    :return: 排序后的文件列表
    """
    # 构建搜索路径
    search_path = os.path.join(directory, pattern)
    
    # 使用 glob 搜索文件
    files = glob.glob(search_path)
    
    # 按文件名排序
    files.sort(key=lambda x: os.path.getctime(x))
    
    return files

def answer(model, conv_mode, qa, tokenizer, image_tensor, args):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qa)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
    
def get_example(questions, questions_featrues, question, filter_model):
    question = clip.tokenize(question).to("cuda:0")
    question_features = filter_model.encode_text(question)
    id = np.random.choice(np.argsort((question_features @ questions_featrues.T).squeeze(0).detach().cpu().numpy())[::-1][2:10])
    return questions[id]
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "ChatUniVi"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    file_list = os.listdir(args.question_file)
    json_files = [filename for filename in file_list if filename.endswith('.json')]
    json_files = get_chunk(json_files, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    count = 0
    
    filter_model, filter_preprocess = clip.load("../MovieChat/ckpt/ViT-B-32.pt", device="cuda:0")
    with open("../MovieChat/Outputs/examples_breakpoint.json", "r") as f:
        examples = json.load(f)
    questions_features = []
    questions = []
    for question in examples.keys():
        questions_features.append(clip.tokenize(question))
        questions.append(question)
    questions_features = torch.cat(questions_features, dim=0).to("cuda:0")
    with torch.no_grad():
        questions_features = filter_model.encode_text(questions_features)

    for file in tqdm(json_files):
        
        if file.endswith('.json'):
            file_path = os.path.join(args.question_file, file)    
            with open(file_path, 'r') as json_file:
                count += 1
                if count > 0:
                    movie_data = json.load(json_file)
                    global_key = movie_data["info"]["video_path"]
                    image_list = get_sorted_files(directory=args.image_folder, pattern=global_key+'*')
                    global_value = []
                    for id, qa_key in enumerate(movie_data["breakpoint"]):
                        qa = qa_key["question"]
                        time = qa_key["time"]
                        image_file = os.path.join(args.image_folder, f"{global_key}_{time}.jpg")
                        
                        try:
                            image = Image.open(image_file)
                        except:
                            image = Image.open(image_list[id])
                            
                        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                        
                        if model.config.mm_use_im_start_end:
                            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                        else:
                            image_token = DEFAULT_IMAGE_TOKEN
                            
                        prompt = image_token + '\n' + "Please describe the image in less than 50 words."

                        outputs = answer(model=model, conv_mode=args.conv_mode, qa=prompt, tokenizer=tokenizer, image_tensor=image_tensor, args=args)
                        
                        example_question = get_example(questions, questions_features, qa, filter_model)
                        example_answer = examples[example_question]
                        example = "You should follow the format of the example: ```Here is the question: " + example_question + "Answer the question in less than 20 words:" + example_answer + '```\n'
                        
                        prompt = example + image_token + '\n' + f"Here is the description of the image:```{outputs}```. \
                            Here is the question:```{qa}``` Please answer the question according to the image and description in less than 20 words."
                            
                        outputs = answer(model=model, conv_mode=args.conv_mode, qa=prompt, tokenizer=tokenizer, image_tensor=image_tensor, args=args)
                        
                        qa_key["pred"] = outputs
                        global_value.append(qa_key)
                        print(f"Question:{qa}\nAnswer:"+qa_key["pred"]+"\n")

                    result_data = {}
                    result_data[global_key] = global_value
                    with open(answers_file, 'a') as output_json_file:
                        output_json_file.write(json.dumps(result_data))
                        output_json_file.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--model_use", type=str, default="BASE")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
