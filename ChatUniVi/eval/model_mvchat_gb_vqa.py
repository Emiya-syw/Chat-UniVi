import argparse
import torch
import os
import json
from tqdm import tqdm
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math
from decord import VideoReader, cpu
import decord
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.editor import *
import subprocess
import random as rnd
import clip 

def video_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)

def read_json(file):
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=224, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.
    video_mask = np.zeros(max_frames, dtype=np.int64)
    max_video_length = 0

    # T x 3 x H x W
    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
        slice_len = patch_images.shape[0]

        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            video[:slice_len, ...] = patch_images

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return torch.from_numpy(video), video_mask

class long_video_slover():
    def __init__(self, fragment_video_path, vis_processor, device):
        self.n_segment = 128
        self.fragment_video_path = fragment_video_path
        self.filter_model, self.filter_preprocess = clip.load("../MovieChat/ckpt/ViT-B-32.pt", device=device)
        self.n_frms = 8
        self.device = device
        self.vis_processor = vis_processor
        
        
    def parse_video_fragment(self, video_path, video_length, n_stage, n_samples):
        decord.bridge.set_bridge("torch")
        per_video_length = video_length / n_samples
        # cut video from per_video_length(n_stage-1, n_stage)
        self.capture_video(video_path, per_video_length, n_stage)
    
    def capture_video(self, video_path, per_video_length, n_stage):
        start_time = n_stage * per_video_length
        end_time = (n_stage+1) * per_video_length
        video = CompositeVideoClip([VideoFileClip(video_path).subclip(start_time,end_time)])
        video.write_videofile(self.fragment_video_path)
        
    def load_video(self, video_path, question, n_frms=8, height=-1, width=-1, sampling="uniform", return_msg = False):
        decord.bridge.set_bridge("torch")
        vr = VideoReader(uri=video_path, height=height, width=width)

        vlen = len(vr)
        start, end = 0, vlen
        # 在预定义的最大帧数和片段帧数之间取最小值
        n_frms = min(n_frms, vlen)

        if sampling == "uniform":
            # 从视频中均匀采样n_frms帧
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
        elif sampling == "headtail":
            indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
            indices = indices_h + indices_t
        elif sampling == "clip":

            interval = 2
            
            indices = np.arange(start, end, vlen / (n_frms/2)).astype(int).tolist() 
            indices_finegrained = np.arange(start, end, interval).astype(int).tolist() 
            
            patch_images = [Image.fromarray(f) for f in vr.get_batch(indices_finegrained).numpy()]

            video_fragment = [self.filter_preprocess(frm).unsqueeze(0) for frm in patch_images]#.to(self.device).permute(1,0,2,3)
            video_fragment = torch.cat(video_fragment, dim=0).to(self.device)
            # print(video_fragment.shape) T 3 224 224 
            tokenize_text = clip.tokenize(question).to(self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = self.filter_model(video_fragment, tokenize_text)
                probs = logits_per_text.softmax(dim=-1).cpu().numpy().reshape(-1)
                indices_question = np.argsort(probs)[::-1][:int(n_frms/2)]
                indices_finegrained = indices_question * interval
                indices.extend(indices_finegrained)
                indices = list(dict.fromkeys(indices))
                    
            indices.sort()
            
        else:
            raise NotImplementedError

        # get_batch -> T, H, W, C
        # temp_frms = vr.get_batch(indices)
        # tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        # frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)
        patch_images = [Image.fromarray(f) for f in vr.get_batch(indices).numpy()]
        frms = torch.stack([self.vis_processor.preprocess(frm, return_tensors='pt')['pixel_values'][0] for frm in patch_images])
        # if not return_msg:
        #     return frms

        # fps = float(vr.get_avg_fps())
        # sec = ", ".join([str(round(f / fps, 1)) for f in indices])
        # # " " should be added in the start and end
        # msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
        return frms
    
    def get_video_features(self, video_path, question):
        video_length = video_duration(video_path)
        video_fragment_list = []
        for i in range(self.n_segment):
            print(i)
            video_fragment = self.parse_video_fragment(video_path, video_length, i, self.n_segment)
            video_fragment = self.load_video(
                    video_path=self.fragment_video_path,
                    question=question,
                    n_frms=self.n_frms, 
                    height=224,
                    width=224,
                    sampling ="clip", 
                    return_msg = True,
                )
            # video_fragment = video_fragment.unsqueeze(0).to(self.device)
            # print(video_fragment.shape)
            # import sys
            # sys.exit(0)
            video_fragment_list.append(video_fragment.to(self.device))
        video_fragment_list = torch.cat(video_fragment_list, dim=0)
        return video_fragment_list, 1
            


def answer(model, conv_mode, qa, tokenizer, video_frames, args):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qa)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
        0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # print(input_ids.shape, video_frames.shape) (1 1035) (979 3 224 224)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_frames.half().cuda(),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    output_ids = output_ids.sequences
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

    if model.config.config["use_cluster"]:
        for n, m in model.named_modules():
            m = m.to(dtype=torch.bfloat16)

    # Load the ground truth file
    file_list = os.listdir(args.question_file)
    json_files = [filename for filename in file_list if filename.endswith('.json')]
    json_files = get_chunk(json_files, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    Video_Solver = long_video_slover(fragment_video_path=f'./results/inter_video_{args.chunk_idx}.mp4', vis_processor=image_processor, device='cuda:0')
    # Iterate over each sample in the ground truth file
    count = 0

    filter_model, filter_preprocess = clip.load("../MovieChat/ckpt/ViT-B-32.pt", device="cuda:0")
    with open("../MovieChat/Outputs/examples_global.json", "r") as f:
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
                    video_path = os.path.join(args.video_folder, global_key)
                    
                    
                    # Check if the video exists
                    # if video_path is not None:  # Modified this line
                    #     if args.max_frames:
                    #         video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=args.max_frames)
                    #     else:
                    #         video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH)
                    global_value = []
                    for id, qa_key in enumerate(movie_data["global"]):
                        qa = qa_key["question"]
                        video_frames, slice_len = Video_Solver.get_video_features(video_path, qa)
                        # try:
                        if model.config.mm_use_im_start_end:
                            video_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN
                        else:
                            video_token = DEFAULT_IMAGE_TOKEN * slice_len
                        
                        prompt = video_token + '\n' + "First, please count the number of fragments in the video. Second, please conclude the fragments in less than 150 words."
                        outputs = answer(model=model, conv_mode=args.conv_mode, qa=prompt, tokenizer=tokenizer, video_frames=video_frames, args=args)
                        
                        example_question = get_example(questions, questions_features, qa, filter_model)
                        example_answer = examples[example_question]
                        example = "You should follow the format of the example: ```Here is the question: " + example_question + "Answer the question in less than 20 words:" + example_answer + '```\n'

                        prompt = example + video_token + '\n' + f"Here is the description of the video:```{outputs}```. \
                            Here is the question:```{qa}``` Please answer the question according to the video and description in less than 20 words."
                            
                        outputs = answer(model=model, conv_mode=args.conv_mode, qa=prompt, tokenizer=tokenizer, video_frames=video_frames, args=args)
                        

                        qa_key["pred"] = outputs
                        global_value.append(qa_key)
                        print(f"Question:{qa}\nAnswer:"+qa_key["pred"]+"\n")
                        # except Exception as e:
                        #     print(f"Error processing video file '{global_key}': {e}")

                    result_data = {}
                    result_data[global_key] = global_value
                    with open(answers_file, 'a') as output_json_file:
                        output_json_file.write(json.dumps(result_data))
                        output_json_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-list", type=str, default="tables/answers_list.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    eval_model(args)
