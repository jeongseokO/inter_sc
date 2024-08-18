import concurrent.futures
from collections import Counter
import re
import pandas as pd
from vllm import LLM, SamplingParams
import flash_attn
from datasets import load_dataset, Dataset
from tqdm import tqdm
import string
import heapq
import argparse
import time
parser = argparse.ArgumentParser(description='model_data_selection')
parser.add_argument('--model', type=str)    
parser.add_argument('--dataset', type=str)

parser.add_argument('--n_tokens_per_iter', type=int)   
parser.add_argument('--max_tokens', type=int)

parser.add_argument('--n_init', type=int)      
parser.add_argument('--n_filter', type=int)
parser.add_argument('--n_output_per_one', type=int)

args = parser.parse_args()
device = "cuda"
print(f"Inter SC_beam_Expand\nNum Path: {args.n_init} + {args.n_filter} x {args.n_output_per_one}, Interval: {args.n_tokens_per_iter}")
def top_k_indices(nums, k):
    # nums 리스트의 값과 인덱스를 튜플로 묶은 리스트 생성
    num_with_indices = [(num, idx) for idx, num in enumerate(nums)]
    
    # 가장 큰 k개의 요소를 찾음 (값을 기준으로)
    largest_k = heapq.nlargest(k, num_with_indices)
    
    # 인덱스만 추출하여 반환
    top_k_indices = [idx for num, idx in largest_k]
    return top_k_indices
def get_messages(context=None, question=None, cot_ex=None, choices=None, model_name=None, dataset=None):
    if model_name  =="mistral_7b":
        if dataset == "mmlu":
            messages = "<s>[INST] You are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet. So for instance the following:"
            for key in cot_ex:
                messages += ("Example Problem " + key + ": " + cot_ex[key])
            messages += "</s>\n[INST]"
            messages += "Solve the following problem: " + question + "\nOptions: A:" + choices[0] + ", B:" + choices[1] + ", C:" + choices[2] + ", D:" + choices[3]  + "\nEnsure your final answer is presented within the format '#### {Alphabet answer}'[/INST]" 
        elif dataset == "mmlu_short":
            messages = "<s>[INST] You are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {answer}' format. So for instance the following:"
            for key in cot_ex:
                messages += ("Example Problem " + key + ": " + cot_ex[key])
            messages += "</s>\n[INST]"
            messages += "Solve the following problem: " + question + "\nEnsure your final answer is presented within the format '#### {answer}'[/INST]" 
        
        elif dataset == "race":
            messages = "<s>[INST] You are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet. So for instance the following:"
            for key in cot_ex:
                messages += ("Example Problem " + key + ": " + cot_ex[key])
            messages += "</s>\n[INST]"
            messages += (context + "\nQuestion: " + question + "\nOptions: A:"+choices[0]+", B:"+choices[1]+ ", C:"+choices[2]+", D:"+choices[3]  + "Ensure your final answer is presented within the format '#### {Alphabet answer}' [/INST]") 
        elif dataset == "aqua":
            messages = "<s>[INST] You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet. So for instance the following:"
            for key in cot_ex:
                messages += ("Example Problem " + key + ": " + cot_ex[key])
            messages += "</s>\n[INST]"
            messages += "Solve the following mathematical problem: " + question + "\nOptions: "+choices[0]+", "+choices[1]+ ", "+choices[2]+", "+choices[3]+", "+choices[4]+" Ensure your final answer is presented within the format '#### {Alphabet answer}' [/INST]"
        elif dataset == "gsm8k":
            messages = "<s>[INST] You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {numeric answer}' format, where the answer is solely a number. So for instance the following:"
            for key in cot_ex:
                messages += ("Example Problem " + key + ": " + cot_ex[key])
            messages += "</s>\n[INST]"
            messages += "Solve the following mathematical problem: " + question  + "Ensure your final answer is presented within the format '#### {numeric answer}' [/INST]"
        elif dataset == "math":
            messages = "<s>[INST] You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {answer}' format. So for instance the following:"
            for key in cot_ex:
                messages += ("Example Problem " + key + ": " + cot_ex[key])
            messages += "</s>\n[INST]"
            messages += "Solve the following mathematical problem: " + question  + "Ensure your final answer is presented within the format '#### {answer}'.[/INST]"
        elif dataset in ["triviaqa","nq"]:
            messages = "<s>[INST] You are a methodical problem solver, adept at answering questions. Conclude your explanation with the answer in a '#### {answer}' format. So for instance the following:"
            for key in cot_ex:
                messages += ("Example Problem " + key + ": " + cot_ex[key])
            messages += "</s>\n[INST]"
            messages += (question  + "Ensure your final answer is presented within the format '#### {answer}'.[/INST]")
        

    elif model_name == "gemma_7b":
        if dataset == "mmlu":
            messages = "<start_of_turn>user\nYou are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet.<end_of_turn>"
            for key in cot_ex:
                messages += ("\n<start_of_turn>user\nExample Problem"  + key + ": " + cot_ex[key]["Question"] + "<end_of_turn>\n<start_of_turn>model\n" + cot_ex[key]["Answer"] + "<end_of_turn>")
            messages += ("\n<start_of_turn>user\n" + question + "\nOptions: A:" + choices[0] + ", B:" + choices[1] + ", C:" + choices[2] + ", D:" + choices[3]  + "\nEnsure your final answer is presented within the format '#### {Alphabet answer}'. Let's think step by step.<end_of_turn>\n<start_of_turn>model")
        elif dataset == "mmlu_short":
            messages = "<start_of_turn>user\nYou are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {answer}' format.<end_of_turn>"
            for key in cot_ex:
                messages += ("\n<start_of_turn>user\nExample Problem"  + key + ": " + cot_ex[key]["Question"] + "<end_of_turn>\n<start_of_turn>model\n" + cot_ex[key]["Answer"] + "<end_of_turn>")
            messages += ("\n<start_of_turn>user\n" + question + "\nEnsure your final answer is presented within the format '#### {answer}'. Let's think step by step.<end_of_turn>\n<start_of_turn>model")
        
        elif dataset == "race":
            messages = "<start_of_turn>user\nYou are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet.<end_of_turn>"
            for key in cot_ex:
                messages += ("\n<start_of_turn>user\nExample Problem"  + key + ": " + cot_ex[key]["Question"] + "<end_of_turn>\n<start_of_turn>model\n" + cot_ex[key]["Answer"] + "<end_of_turn>")
            messages += ("\n<start_of_turn>user\n" + context + "\nQuestion: " + question + "\nOptions: A:" + choices[0] + ", B:" + choices[1] + ", C:" + choices[2] + ", D:" + choices[3]  + "\nEnsure your final answer is presented within the format '#### {Alphabet answer}'. Let's think step by step.<end_of_turn>\n<start_of_turn>model")
        
        elif dataset == "aqua":
            messages = "<start_of_turn>user\nYou are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet.<end_of_turn>"
            for key in cot_ex:
                messages += ("\n<start_of_turn>user\nExample Problem"  + key + ": " + cot_ex[key]["Question"] + "<end_of_turn>\n<start_of_turn>model\n" + cot_ex[key]["Answer"] + "<end_of_turn>")
            messages += ("\n<start_of_turn>user\n"+ question + "\nOptions: "+choices[0]+", "+choices[1]+ ", "+choices[2]+", "+choices[3]+", "+choices[4]+" Ensure your final answer is presented within the format '#### {Alphabet answer}'. Let's think step by step.<end_of_turn>\n<start_of_turn>model")
        
        elif dataset == "gsm8k":
            messages = "<start_of_turn>user\nYou are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {numeric answer}' format, where the answer is solely a number.<end_of_turn>"
            for key in cot_ex:
                messages += ("\n<start_of_turn>user\nExample Problem"  + key + ": " + cot_ex[key]["Question"] + "<end_of_turn>\n<start_of_turn>model\n" + cot_ex[key]["Answer"] + "<end_of_turn>")
            messages += ("\n<start_of_turn>user\n" + question  + "Ensure your final answer is presented within the format '#### {numeric answer}'. Let's think step by step.<end_of_turn>\n<start_of_turn>model")
        elif dataset == "math":
            messages = "<start_of_turn>user\nYou are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {answer}' format.<end_of_turn>"
            for key in cot_ex:
                messages += ("\n<start_of_turn>user\nExample Problem"  + key + ": " + cot_ex[key]["Question"] + "<end_of_turn>\n<start_of_turn>model\n" + cot_ex[key]["Answer"] + "<end_of_turn>")
            messages += ("\n<start_of_turn>user\n" + question  + "Ensure your final answer is presented within the format '#### {answer}'. Let's think step by step.<end_of_turn>\n<start_of_turn>model")
        elif dataset in ["triviaqa","nq"]:
            messages = "<start_of_turn>user\nYou are a methodical problem solver, adept at answering questions. Conclude your explanation with the answer in a '#### {answer}' format.<end_of_turn>"
            for key in cot_ex:
                messages += ("\n<start_of_turn>user\nExample Problem"  + key + ": " + cot_ex[key]["Question"] + "<end_of_turn>\n<start_of_turn>model\n" + cot_ex[key]["Answer"] + "<end_of_turn>")
            messages += ("\n<start_of_turn>user\n" + question  + "Ensure your final answer is presented within the format '#### {answer}'. Let's think step by step.<end_of_turn>\n<start_of_turn>model")

    else:           
        if dataset == "mmlu":
            messages = [
                {"role": "system", "content": "You are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet"},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
            
            messages.append({"role": "user", "content": "Solve the following problem: " + question + "\nOptions: A:" + choices[0] + ", B:" + choices[1] + ", C:" + choices[2] + ", D:" + choices[3]  + "\nEnsure your final answer is presented within the format '#### {Alphabet answer}'."})
        elif dataset == "mmlu_short":
            messages = [
                {"role": "system", "content": "You are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {answer}' format"},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
            
            messages.append({"role": "user", "content": "Solve the following problem: " + question + "\nEnsure your final answer is presented within the format '#### {answer}'."})
        
        elif dataset == "race" and context: 
            messages = [
            {"role": "system", "content": "You are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet"},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
            
            messages.append({"role": "user", "content": context + "\nQuestion: : " + question + "\nOptions: A:"+choices[0]+", B:"+choices[1]+ ", C:"+choices[2]+", D:"+choices[3]  + "Ensure your final answer is presented within the format '#### {Alphabet answer}'."})
        
        elif dataset == "aqua":
            messages = [
            {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet"},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
            
            messages.append({"role": "user", "content": "Solve the following mathematical problem: " + question + "\nOptions: "+choices[0]+", "+choices[1]+ ", "+choices[2]+", "+choices[3]+", "+choices[4]+" Ensure your final answer is presented within the format '#### {Alphabet answer}'."})
        
        elif dataset == "gsm8k":
            messages = [
            {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {numeric answer}' format, where the answer is solely a number."},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
            
            messages.append({"role": "user", "content": "Solve the following mathematical problem: " + question  + " Ensure your final answer is presented within the format '#### {numeric answer}'."})

        elif dataset == "math":
            messages = [
            {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {answer}' format."},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
            
            messages.append({"role": "user", "content": "Solve the following mathematical problem: " + question  + "Ensure your final answer is presented within the format '#### {answer}'."})
        
        elif dataset in ["triviaqa","nq"]:
            messages = [
            {"role": "system", "content": "You are a methodical problem solver, adept at answering questions. Conclude your explanation with the answer in a '#### {answer}' format."},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
            
            messages.append({"role": "user", "content": question  + "Ensure your final answer is presented within the format '#### {answer}'"})
        
    return messages

def get_added_prompt(prompt, added_string):
    return (prompt + added_string)

def get_completion(context=None, question=None, choices=None, cot_ex=None, model=None, model_name=None, dataset=None):
    start_time = time.perf_counter()
    messages = get_messages(context, question, cot_ex, choices, model_name, dataset)
    tokenizer = model.get_tokenizer()
    if model_name in ["gemma_7b", "mistral_7b"]:
        conversations = [messages],
        if isinstance(conversations, tuple):
            conversations = conversations[0]
    else:
        conversations = tokenizer.apply_chat_template(
            [messages],
            tokenize=False,
        )

    def generate_and_filter(init, conversations, n_generate, n_filter):
        s = time.perf_counter()
        outputs = model.generate(
            conversations,
            SamplingParams(
                n=n_generate,
                temperature=0.7,
                top_p=0.8,
                max_tokens=args.n_tokens_per_iter,
                stop_token_ids=[tokenizer.eos_token_id],
                skip_special_tokens=False,
                logprobs=1
            )
        )
        e = time.perf_counter()
        print(f"Generation job: {e - s} sec")

        generated_texts = [text.text for output in outputs for text in output.outputs]
        log_probs = [text.cumulative_logprob for output in outputs for text in output.outputs]

        top_k_indices = sorted(range(len(log_probs)), key=lambda i: log_probs[i], reverse=True)[:n_filter]
        if init:
            new_convs = [get_added_prompt(conversations[id//n_generate], generated_texts[id]) for id in range(n_generate)]
        else:
            new_convs = [get_added_prompt(conversations[id//n_generate], generated_texts[id]) for id in range(n_generate * n_filter)]

        top_k_conversations = [new_convs[i] for i in top_k_indices]


        return new_convs, top_k_indices, top_k_conversations

    n_initial = args.n_init #10
    n_intermediate = args.n_filter #5
    n_output_per_one = args.n_output_per_one #2
    max_iterations = args.max_tokens // args.n_tokens_per_iter

    # Initial generation of 10 completions
    current_conversations = conversations
    prompt_length = len(conversations[0])
    generated_texts, top_k_indices, top_k_completions = generate_and_filter(True, current_conversations, n_initial, n_intermediate)
    
    new_conversations = top_k_completions
    #print(f"all completions: {new_conversations[0]}")
    
    for _ in range(max_iterations):
        
        #print(f"All Completions: {all_completions}")
        # Generate 2 completions for each of the 5 selected completions, resulting in 10 completions
        generated_texts, top_k_indices, further_completions = generate_and_filter(False, new_conversations, n_output_per_one, n_intermediate)
        n_output_per_one += 1
        #print(f"Further Comp: {further_completions}")
        #print(f"Number of conversations: {len(new_conversations)}")
        if all("####" in comp[prompt_length:] for comp in generated_texts):
            #print(f"EOS Token: {tokenizer.eos_token_id}")
            print("Found EoS token, Break")
            new_conversations = generated_texts
            #print(f"completions: {new_conversations[0]}")
            break
        else:
            new_conversations = further_completions

    end_time = time.perf_counter()
    print(f">> Processing time: {end_time - start_time}")
    
    return new_conversations  # Return the top 10 completions



def extract_alphabet(text):
    pattern = r'#{4}\s*([A-Z])'
    matches = re.findall(pattern, text)
    if matches:
        answer = matches[-1].strip()
        if answer in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
            return answer
    return "No Match"
            
def extract_number(text):
    """
    Extracts the last number following '####' in the input text.
    The number may contain commas, periods, and a sign (positive or negative).
    """
    pattern = r'#### ([\d,.,-]+)'
    matches = re.findall(pattern, text)

    if matches:
        number_str = matches[-1].replace(',', '')  # 쉼표 제거
        # 마지막에 마침표가 있는 경우 이를 제거
        if number_str.endswith('.'):
            number_str = number_str[:-1]
        return number_str
    return "No Match"
def extract_string(text):
    # #### 뒤에 오는 숫자 또는 문자를 찾는 정규 표현식
    pattern = r'#### ([\w\\{}]+)'

    # 정규 표현식과 일치하는 부분을 찾기
    match = re.search(pattern, text)
    
    # 일치하는 부분이 있다면, 결과를 반환
    if match:
        return match.group(1)
    else:
        return "No Match"
def extract_triviaqa(text):
    # #### 뒤에 오는 텍스트를 찾는 정규 표현식
    pattern = r'#{4}\s*(.*)'

    # 정규 표현식과 일치하는 부분을 찾기
    match = re.search(pattern, text)
    
    # 일치하는 부분이 있다면, 텍스트를 반환
    if match:
        return match.group(1).strip().replace(".", "")
    else:
        return "No Match"
def safe_convert_to_float(value):
    try:
        # 문자열을 정수로 변환 시도
        return float(value)
    except ValueError:
        # 변환할 수 없는 경우 원래 값을 반환
        return value
def extract_latex_boxed_content(latex_str):
    """
    Extracts the content within \boxed or \\boxed commands in a given LaTeX string.
    This function handles both single and double backslashes before 'boxed'.
    """
    # Initialize an empty stack to track the depth of curly braces
    stack = []
    # Buffer to hold the extracted content
    content = []
    # Flag to indicate if we are inside the \boxed or \\boxed command
    inside_boxed = False
    # Iterate through the string
    i = 0
    while i < len(latex_str):
        # Check for \boxed or \\boxed command start
        if latex_str[i:i+7] == '\\boxed{' or latex_str[i:i+8] == '\\\\boxed{':
            if not inside_boxed:  # If we're not already inside a \boxed command
                inside_boxed = True
                # Move the index to the end of the matched command
                i += 7 if latex_str[i:i+7] == '\\boxed{' else 8
                continue  # Skip to the next iteration
        # Check for the closing brace of the \boxed or \\boxed command
        elif latex_str[i] == '}' and inside_boxed:
            if stack:
                # If there's an open brace in the stack, pop it
                stack.pop()
            else:
                # If there are no open braces, we've reached the end of the \boxed content
                inside_boxed = False
                return ''.join(content)  # Return the extracted content
        # If we're inside the \boxed command, add characters to the content buffer
        if inside_boxed:
            if latex_str[i] == '{':
                # If we encounter an open brace, add it to the stack
                stack.append('{')
            content.append(latex_str[i])  # Add the current character to the content buffer
        i += 1  # Move to the next character

    # If the loop ends without returning, it means we didn't find a matching closing brace
    return ''.join(content) if inside_boxed else "No Match"
def process_completion(answers, real_answer, dataset):
    num = []
    if dataset in ["mmlu", "race", "aqua"]:
        for answer in answers:
            extracted_ans = extract_alphabet(answer)
            num.append(extracted_ans)
    elif dataset in ["gsm8k"]:
        for answer in answers:
            extracted_ans = extract_number(answer)
            num.append(extracted_ans)
    elif dataset in ["math"]:
        for answer in answers:
            extracted_ans = extract_string(answer)
            num.append(extracted_ans)
    elif dataset in ["triviaqa", "nq", "mmlu_short"]:
        for answer in answers:
            extracted_ans = extract_triviaqa(answer)
            num.append(extracted_ans)
    # Majority vote
    count = Counter(num)
    hard_ratio = sum([real_answer in i for i in num])
    hard_ratio = hard_ratio/len(num)
    # 가장 많이 나온 답과 그 개수 찾기
    most_common = count.most_common(1)[0][0]
    most_count = count.most_common(1)[0][1]

    # "Not a Number"가 가장 많이 나온 경우 두 번째로 많이 나온 답 찾기
    if most_common == "No Match" and len(count) > 1:
        most_common = count.most_common(2)[1][0]
        most_count = count.most_common(2)[1][1]
    majority_rate = most_count/len(num)
    # 결과 반환
    matching_indexes = [index for index, item in enumerate(num) if item == most_common]
    index = num.index(most_common)
    #print(matching_indexes)
    return most_common, majority_rate, hard_ratio, index, num



def parallel_process(context=None, question=None, answer=None, option=None, cot_ex=None, model=None, model_name=None, dataset=None):
    if option is None:
        option = [None for i in range(1000)]
    if context is None:
        context = [None for i in range(1000)]
    results = []
    for (c, q, a, o) in zip(context, question, answer, option):
        if isinstance(a, int) and dataset not in ["gsm8k", "math", "triviaqa", "nq", "mmlu_short"]:
            real_answer = list(string.ascii_uppercase)[a]
            extracted_real_answer = real_answer
        elif dataset in ["triviaqa"]:
            real_answer = a['value']
            extracted_real_answer = a['value']
            extracted_real_alias = a['aliases']
        elif dataset in ["nq"]:
            real_answer = a[0]
            extracted_real_answer = a[0]
            extracted_real_alias = a
        elif dataset in ["mmlu_short"]:
            #print(o)
            real_answer = o[a]
            extracted_real_answer = real_answer
        else:
            real_answer = a
            extracted_real_answer = a
            if dataset in ["gsm8k"]:
                extracted_real_answer = extract_number(real_answer)
            elif dataset in ["math"]:
                extracted_real_answer = extract_string(real_answer)
        original_answer_completion = get_completion(context=c, question=q, choices=o, cot_ex=cot_ex, model=model, model_name=model_name, dataset=dataset)
        original_most_common, original_conf, original_hr, original_index, ans_list = process_completion(original_answer_completion, extracted_real_answer, dataset)
        original_rp = original_answer_completion
        
        if dataset in ["triviaqa","nq"]:
            original_is_correct = safe_convert_to_float(original_most_common) in safe_convert_to_float(extracted_real_alias)
        else:
            original_is_correct = safe_convert_to_float(original_most_common.replace(',','')) == safe_convert_to_float(extracted_real_answer.replace(',',''))
        print(f">> Answer lists: {ans_list}")
        print(f">> Original Correct: {original_most_common}, {original_is_correct}")
        print(f">> Golden    Answer: {extracted_real_answer}")
        results.append((q, real_answer, original_most_common, original_is_correct, original_conf, original_hr, original_rp))
    return results
def get_cot_Ex(dataset, model_name):
    cot_ex = {}
    if model_name == "gemma_7b":
        if dataset == "mmlu":
            cot_ex['example1'] = {
                "Question": "What is the value of p in 24 = 2p?\nOptions: A:p=4, B:p=8, C:p=12, D:p=24",
                "Answer": "To find the value of p in the equation 24 = 2p, we need to solve for p step by step. First, we start with the given equation, which is 24 = 2p. To isolate p, we divide both sides of the equation by 2. This gives us 24 ÷ 2 = 2p ÷ 2. Simplifying both sides, we get 12 = p. Therefore, the value of p is 12. Thus, the correct answer is option #### C"
            }
            cot_ex['example2'] = {
                "Question": "Ms. Perez drove a total of 40 miles in 5 days. She drove the same number of miles each day. How many miles did Ms. Perez drive each day?\nOptions: A:5, B:7, C:8, D:9",
                "Answer": "Ms. Perez drove a total of 40 miles over the course of 5 days. Since she drove the same number of miles each day, we can determine how many miles she drove each day by dividing the total miles by the number of days. Dividing 40 miles by 5 days gives us 8 miles per day. Therefore, the answer is option #### C"
            }
            cot_ex['example3'] = {
                "Question": "What is the remainder of 21 divided by 7?\nOptions: A:21, B:7, C:1, D:None of these",
                "Answer": "To find the remainder when 21 is divided by 7, we divide 21 by 7. The result of the division is 3, which is an integer, meaning there is no remainder. Thus, when 21 is divided by 7, it divides evenly with no remainder left over. Therefore, the remainder of 21 divided by 7 is 0. The answer is #### D"
            }
            cot_ex['example4'] = {
                "Question": "Identify the algebraic expression that matches the word phrase: 4 times as many cans as Tom collected.\nOptions: A:4t, B:t over 4, C:t - 4, D:t + 4",
                "Answer": "To identify the algebraic expression that matches the phrase '4 times as many cans as Tom collected,' we start by letting x represent the number of cans Tom collected. The phrase '4 times as many' means that we need to multiply the number of cans Tom collected by 4. Therefore, the algebraic expression that represents '4 times as many cans as Tom collected' is 4x. The answer is #### A"
            }
            cot_ex['example5'] = {
                "Question": "What is the value of the expression 2(3(4^2 + 1)) - 2^3?\nOptions: A:156, B:110, C:94, D:48",
                "Answer": "To find the value of the expression 2(3(4^2 + 1)) - 2^3, we need to evaluate it step by step in a detailed manner: First, we calculate the exponent 4^2: 4^2 = 16. Next, we add 1 to this result: 16 + 1 = 17. Then, we multiply this result by 3: 3 × 17 = 51. After that, we multiply the result by 2: 2 × 51 = 102. Next, we evaluate the exponent 2^3: 2^3 = 8. Finally, we subtract this result from 102: 102 - 8 = 94. Thus, the value of the expression 2(3(4^2 + 1)) - 2^3 is 94. The answer is option #### C"
            }
            cot_ex['example6'] = {
                "Question": "Compute 22 / 2 + 9.\nOptions: A:10, B:11, C:20, D:2",
                "Answer": "To compute the expression 22 / 2 + 9, we start by performing the division first. Dividing 22 by 2 gives us 11. Next, we add 9 to the result of the division. Therefore, 11 plus 9 equals 20. Thus, the value of the expression 22 / 2 + 9 is 20. The answer is #### C"
            }
            cot_ex['example7'] = {
                "Question": "What is the least common multiple of 4 and 10?\nOptions: A:14, B:20, C:40, D:60",
                "Answer": "To find the least common multiple (LCM) of 4 and 10, we follow these steps in a detailed manner: First, we determine the prime factors of each number. The prime factors of 4 are 2 and 2, which means 4 = 2 × 2 = 2^2. The prime factors of 10 are 2 and 5, which means 10 = 2 × 5. Next, we identify the highest power of each prime factor that appears in either number. For the prime number 2, the highest power that appears is 2^2 (from the number 4). For the prime number 5, the highest power that appears is 5 (from the number 10). We then multiply these highest powers together to find the LCM. We multiply 2^2 and 5 together: 2^2 × 5 = 4 × 5 = 20. Therefore, the least common multiple of 4 and 10 is 20. The answer is #### B"
            }
            cot_ex['example8'] = {
                "Question": "Estimate 711 + 497. The sum is between which numbers?\nOptions: A:50 and 400, B:450 and 700, C:750 and 1000, D:1,050 and 1,300",
                "Answer": "To estimate the sum of 711 and 497, let’s round each number to the nearest hundred and then add them together. 711 rounds to 700 and 497 rounds to 500. Now, add the rounded numbers: 700 + 500 = 1200. The sum of 711 and 497 is close to 1200, which lies between 1050 and 1300. Therefore, the answer is #### D"
            }
        elif dataset == "mmlu_short":
            cot_ex['example1'] = {
                "Question": "What is the value of p in 24 = 2p?",
                "Answer": "To find the value of p in the equation 24 = 2p, divide both sides by 2: 24 ÷ 2 = 12. Therefore, the value of p is 12. The answer is #### 12."
            }

            cot_ex['example2'] = {
                "Question": "Ms. Perez drove a total of 40 miles in 5 days. She drove the same number of miles each day. How many miles did Ms. Perez drive each day?",
                "Answer": "Ms. Perez drove a total of 40 miles over 5 days. Dividing 40 miles by 5 days gives us 8 miles per day. Therefore, the answer is #### 8."
            }

            cot_ex['example3'] = {
                "Question": "What is the remainder of 21 divided by 7?",
                "Answer": "21 ÷ 7 = 3 with no remainder. Therefore, the remainder is 0. The answer is #### 0."
            }

            cot_ex['example4'] = {
                "Question": "Identify the algebraic expression that matches the word phrase: 4 times as many cans as Tom collected.",
                "Answer": "Let x represent the number of cans Tom collected. '4 times as many' means 4x. Therefore, the answer is #### 4x."
            }

            cot_ex['example5'] = {
                "Question": "What is the value of the expression 2(3(4^2 + 1)) - 2^3?",
                "Answer": "Calculate step by step: 4^2 = 16, 16 + 1 = 17, 3 × 17 = 51, 2 × 51 = 102, 2^3 = 8, 102 - 8 = 94. Thus, the value is 94. The answer is #### 94."
            }

            cot_ex['example6'] = {
                "Question": "Compute 22 / 2 + 9.",
                "Answer": "22 ÷ 2 = 11, 11 + 9 = 20. Therefore, the answer is #### 20."
            }

            cot_ex['example7'] = {
                "Question": "What is the least common multiple of 4 and 10?",
                "Answer": "Prime factors of 4 are 2^2 and of 10 are 2 and 5. LCM is 2^2 × 5 = 4 × 5 = 20. Therefore, the answer is #### 20."
            }

            cot_ex['example8'] = {
                "Question": "Estimate 711 + 497. The sum is between which numbers?",
                "Answer": "Round 711 to 700 and 497 to 500. 700 + 500 = 1200. Therefore, the sum is between 1050 and 1300. The answer is #### 1200."
            }
        elif dataset == "aqua":
            cot_ex['example1'] = {
                "Question": """A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower?\nOptions: A)5(√3 + 1), B)6(√3 + √2), C)7(√3 – 1), D)8(√3 – 2), E)None of these""",
                "Answer": """Let the height of the building be h. Initially, he was at an angle of 450. tan 45 = h/distance between car and tower. h = distance between car and tower (since tan 45 = 1). Now, after 10 minutes, it travelled a certain distance, and angle changed to 600. tan 60 = h/x x = h/√3 So, in 10 minutes, it has travelled a distance of h – x = h - h/√3. 10 minutes = h *( 1 – 1√3) h can be travelled in 10 / (1 – 1√3). To travel a distance of x, which is h/√3, it takes : h = 10 / (1 – 1/√3) h / √3 = 10/ √3 * (1 – 1/√3). Multiply numerator and denominator by 1 + √3 ( conjugate of 1 - √3). We get, x = h/√3 = 10 (1 + √3) / 2 = 5* (1 + √3) So, it takes 5(1 + √3) minutes to reach the base of the tower. Answer: #### A"""
            }
            cot_ex['example2'] = {
                "Question": """Find out which of the following values is the multiple of X, if it is divisible by 9 and 12?\nOptions: A)36, B)15, C)17, D)5, E)7""",
                "Answer": """9=3*3 12=3*4 The number should definitely have these factors 3*3*4 36 is the number that has these factors So, 36 is the multiple of X Answer is #### A"""
            }
            cot_ex['example3'] = {
                "Question": """A trader sold an article at a profit of 20% for Rs.360. What is the cost price of the article?\nOptions: A)270, B)300, C)280, D)320, E)315""",
                "Answer": """Cost Price = Selling Price / (100+Profit%) × 100 => 360 / (100+20) × 100 => 360 / 120 × 100 = Rs.300 Option #### B"""
            }
            cot_ex['example4'] = {
                "Question": """Find the total no. of distinct bike no.'s that can be formed using 2 letters followed by 2 no.'s. How many letters need to be distinct?\nOptions: A)74453, B)64543, C)74325, D)65000, E)97656""",
                "Answer": """Out of 26 alphabets two distinct letters can be chosen in 26P2 ways. Coming to the numbers part, there are 10 ways to choose the first digit and similarly, there are another 10 ways to choose the second digit. Hence, there are in total 10X10 = 100 ways. Combined with letters there are 6P2 X 100 ways = 65000 ways to choose vehicle numbers. #### D"""
            }
            cot_ex['example5'] = {
                "Question": """[5 + ? × 19 - 15 - 7]/[13 × 13 - 156] = 6\nOptions: A)4, B)4.5, C)5, D)5.5, E)6.5""",
                "Answer": """(? × 19 - 17)/(169 - 156) = 6 => ? × 19 - 17 = 13 × 6 = 76 => ? × 19 = 78 + 17 = 95 ? = 95/19 = 5 Answer: Option #### C"""
            }
            cot_ex['example6'] = {
                "Question": """A book was sold for Rs 27.50 with a profit of 10%. If it were sold for Rs. 25.75, then would have been percentage of profit and loss ?\nOptions: A)2% Profit, B)3% Profit, C)2% Loss, D)3% Loss, E)4% Loss""",
                "Answer": """S.P.=(100+gain%100∗C.P) So, C.P. = (100/110∗25.75) When S.P. = 25.75 then Profit=25.75−25=Re.0.75 Profit%=0.75/25∗100=3% Answer is #### B"""
            }
            cot_ex['example7'] = {
                "Question": """900 + 5 × 12 = ?\nOptions: A)820, B)202, C)420, D)209, E)960""",
                "Answer": """900 + 5 × 12 = ? or, ? = 900 + 60 = 960 Answer #### E"""
            }
            cot_ex['example8'] = {
                "Question": """What is the units digit of 9^3-7?\nOptions: A)1, B)3, C)5, D)2, E)4""",
                "Answer": """The unit's digit of 9^3 = 9 9-7=2 Answer #### D"""
            }
        elif dataset == "race":
            cot_ex['example1'] = {
                "Question":"""I am a psychologist. I first met Timothy, a quiet, overweight eleven-year-old boy, when his mother brought him to me to discuss his declining grades. A few minutes with Timothy were enough to confirm that his self-esteem and general happiness were falling right along with _ . I asked about Timothy's typical day. He awoke every morning at six thirty so he could reach his school by eight and arrived home around four thirty each afternoon. He then had a quick snack, followed by either a piano lesson or a lesson with his math tutor. He finished dinner at 7 pm, and then he sat down to do homework for two to three hours. Quickly doing the math in my head, I found that Timothy spent an average of thirteen hours a day at a writing desk. What if Timothy spent thirteen hours a day at a sewing machine instead of a desk? We would immediately be shocked, because that would be called children being horribly mistreated. Timothy was far from being mistreated, but the mountain of homework he faced daily resulted in a similar consequence --he was being robbed of his childhood. In fact, Timothy had no time to do anything he truly enjoyed, such as playing video games, watching movies, or playing board games with his friends. Play, however, is a crucial part of healthy child development. It affects children's creativity, their social skills, and even their brain development. The absence of play, physical exercise, and freefrom social interaction takes a serious toll on many children. It can also cause significant health problems like childhood obesity, sleep problems and depression. Experts in the field recommend the minutes children spend on their homework should be no more than ten times the number of their grade level. As a fifthgrader, Timothy should have no more than fifty minutes a day of homework (instead of three times that amount). Having an extra two hours an evening to play, relax, or see a friend would soundly benefit any child's life quality.\nQuestion: What did the writer think of Timothy after learning about his typical day?\nOptions: A: "Timothy was very hardworking.", B: "Timothy was being mistreated.", C: "Timothy had a heavy burden.", D: "Timothy was enjoying his childhood." """,
                "Answer": """Observation: Timothy's grades, self-esteem, and happiness were declining. Daily Routine: Timothy spends around thirteen hours a day working at a desk with school, lessons, and homework. Writer’s Comparison: The writer compares this to child mistreatment, highlighting the excessive burden. Health and Development: Lack of playtime affects creativity, social skills, brain development, and leads to health issues like obesity and depression. Expert Advice: Homework should be limited to fifty minutes for a fifth-grader, not the excessive amount Timothy has. The writer thinks Timothy has a heavy burden. So, the correct answer is: #### C"""
            }
            cot_ex['example2'] = {
                "Question": """From self-driving cars to carebots (care+robots) for elderly people, rapid development in technology has long represented a possible threat to many jobs normally performed by people. But experts now believe that almost 50 percent of occupations existing today will be completely unnecessary by 2025 as artificial intelligence continues to change businesses. "The next fifteen years will see a revolution in how we work, and a revolution will necessarily take place in how we plan and think about workplaces," said Peter Andrew, Director of Workplace Strategy for CBRE Asia Pacific. A growing number of jobs in the future will require creative intelligence, social skills and the ability to use artificial intelligence. The report is based on interviews with 200 experts, business leaders and young people from Asia Pacific, Europe and North America. It shows that in the US technology already destroys more jobs than it creates. But the report states: "Losing occupations does not necessarily mean losing jobs -- just changing what people do." Growth in new jobs could occur as much, according to the research. "The growth of 20 to 40 person companies that have the speed and technological know-how will directly challenge big companies," it states. A 2014 report by Pew Research found 52 percent of experts in artificial intelligence and robotics were optimistic about the future and believed there would still be enough jobs in the next few years. The optimists pictured "a future in which robots do not take the place of more jobs than they create," according to Aaron Smith, the report's co-author. "Technology will continue to affect jobs, but more jobs seem likely to be created. Although there have always been unemployed people, when we reached a few billion people there were billions of jobs. There is no shortage of things that need to be done and that will not change," Microsoft's Jonathan Grudin told researchers.\nQuestion: According to the report, _ . \nOptions:  A: "people won't necessarily lose jobs", B: "big companies will face fewer challenges", C: "small companies will win against big companies", D: "most people will become interested in technology" """, 
                "Answer": """Let's break down the information provided to determine the correct answer. Technology is rapidly developing and poses a threat to many jobs traditionally performed by people. Experts believe that almost 50% of current occupations will be unnecessary by 2025 due to artificial intelligence. The future will demand jobs requiring creative intelligence, social skills, and the ability to use AI. Technology is already destroying more jobs than it creates in the US. However, losing occupations doesn't necessarily mean losing jobs, but rather changing what people do. Growth is expected in small, agile companies with technological expertise. 52% of experts believe there will still be enough jobs in the future. There is a belief that robots will not replace more jobs than they create. Despite technological impacts, there will always be things that need to be done, hence job creation will continue.From this breakdown, the statement that matches the report's findings is: A: "people won't necessarily lose jobs. So, the correct answer is: #### A"""
            }
            cot_ex['example3'] = {
                "Question":"""If you want to be a volunteer, you have to answer the following typical questions. Do you want to work with people, animals or machines? Do you want to work indoors or outdoors, directly serve people in need or serve people behind the scenes? Every year, thousands of people in the west offer volunteer service. Volunteering greatly strengthens the community because it helps the old, the young, the weak, the sick, and the disabled and the injured to solve problems. Volunteers usually help in many different ways. They may give people advice, offer friendship to the young, drive the elderly to church (if up to the driving age), advise kids against drugs, work as assistants in schools or nursing homes, raise funds, plant trees, help out in local libraries and do many other things. Volunteering can be a few hours a week or a few hours a month. Anybody who wants to serve people in need can become a volunteer. In fact, the art of volunteering is a process of both giving and receiving. Volunteering allows volunteers to meet new people, make new friends and mix with people from all walks of life. Volunteering is an excellent way to experiment and try out new techniques and skills, discover your individual talents and explore career choice. Being a volunteer will take you on a wonderful journey and help you learn more than what you can get from books.\nQuestion: What's the best title of the passage?\nOptions: A: "How to hunt for jobs", B: "Volunteer service in the west", C: "How to make friends", D: "How to work with animals" """,
                "Answer": """The passage discusses various aspects of volunteering, including the types of volunteer work, the benefits to the community, and the personal benefits to the volunteers. The focus is on volunteer service in the West and its impact. Given this information, the best title for the passage is: B: "Volunteer service in the west" So, the correct answer is: #### B"""
            }
            cot_ex['example4'] = {
                "Question":"""Let us take a serious look at what the results might be if such a suggestion were accepted. Families might use the time for a real family hour. They might sit around together after dinner and actually talk to one another. It is well known that many of our problems-everything, in fact, from the generation gap to the high divorce rate to some forms of mental illnesses-are caused at least in part by the failure to communicate. We do not tell each other what is disturbing us. The result is emotional difficulty of one kind or another. By using the quiet family hour to discuss our problems, we might get to know each other better, and to like each other better. On evenings when such talk is unnecessary, families could rediscover more active entertainments. Freed from TV, forced to find their own activities, they might take a ride together to watch the sunset. Or they might take a walk together and see the neighborhood with fresh, new eyes. With free time and no TV, children and adults might rediscover reading. There is more entertainment in a good book than in a month of typical TV programming. Educators report that the generation growing up with television can hardly write an English sentence, even at the college level. Writing is often learned from reading. A more cultured new generation could be a product of the quiet hour. At first glance, the idea of an hour without TV seems radical . What will parents do without the electronic baby-sitter? How will we spend the time? But it's not radical at all. It has been only twenty-five years since television came to control our free time. Those of us thirty-five and older can remember childhoods without television, spent partly with radio-which at least involved the listener's imagination-but also with reading, learning, talking, playing games and inventing new activities. It wasn't that difficult, honest. The truth is that we had a ball.\nQuestion: According to the text, good books _ .\nOptions: A: "offer as much knowledge as TV does", B: "are more entertaining than TV", C: "should be made into films", D: "cannot be found easily" """,
                "Answer": """According to the text, good books are more entertaining than TV. So, the correct answer is: B: "are more entertaining than TV. The answer is #### B"""
            }
        elif dataset == "gsm8k":
            cot_ex['example1'] = {
                "Question":"There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "Answer": "We start with 15 trees. Later we have 21 trees.The difference must be the number of trees they planted. So, they must have planted 21-15=6 trees.The answer is #### 6."
            }
            cot_ex['example2'] = {
                "Question":"If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "Answer": "There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is #### 5."
            }
            cot_ex['example3'] = {
                "Question":"Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "Answer": "Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is #### 39."
            }
            cot_ex['example4'] = {
                "Question":"Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                "Answer": "Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is #### 8."
            }
            cot_ex['example5'] = {
                "Question":"Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                "Answer": "He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is #### 9."
            }
            cot_ex['example6'] = {
                "Question":"There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?",
                "Answer": "There are 4 days from Monday to Thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is #### 29."
            }
            cot_ex['example7'] = {
                "Question":"Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?",
                "Answer": "Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is #### 33."
            }
            cot_ex['example8'] = {
                "Question":"Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                "Answer": "She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is #### 8."
            }
        elif dataset == "math":
            cot_ex['example1'] = {
                "Question":"Example Problem: Evaluate $\\lfloor 14.6 \\rfloor-\\lceil-14.6\\rceil$.",
                "Answer": "The greatest integer less than $14.6$ is $14$. The smallest integer greater than $-14.6$ is $-14$. Therefore, the equation can be rewritten as $14-(-14)$, or #### 28."
            }
            cot_ex['example2'] = {
                "Question":"Example Problem: What is the sum of all numbers $a$ for which the graph of $y=x^2+a$ and the graph of $y=ax$ intersect one time?",
                "Answer": "If these two graphs intersect then the point of intersection occur when  \\[x^2+a=ax,\\] or  \\[x^2-ax+a=0.\\] This quadratic has one solution exactly when the discriminant is equal to zero: \\[(-a)^2-4\\cdot1\\cdot a=0.\\] This simplifies to  \\[a(a-4)=0.\\]\n\nThere are exactly two values of $a$ for which the line and parabola intersect one time, namely $a=0$ and $a=4$.  The sum of these values is #### 4"
            }
            cot_ex['example3'] = {
                "Question":"Example Problem: At 50 miles per hour, how far would a car travel in $2\\frac{3}{4}$ hours? Express your answer as a mixed number.",
                "Answer": "In two hours, a car traveling at $50$ mph would travel $50$ mph $\\times 2$ hours $= 100$ miles. Now we find how far a car can travel in $3/4$ of an hour which is $50$ mph $\\times \\frac{3}{4}$ hours $ = \\frac{150}{4} = 37 \\frac{1}{2}$ miles. Thus, the car would travel a total of $100 + 37 \\frac{1}{2}= 137 \\frac{1}{2}$ miles. The answer is #### 137 \\frac{1}{2}"
            }
            cot_ex['example4'] = {
                "Question":"Example Problem: In a rectangular coordinate system, the line $3y = x$ intersects the line $2x + 5y = 11$ at point $A$. What is the sum of the coordinates of point $A$?",
                "Answer": "If the coordinates of point $A$ are $(x,y)$ then $x$ and $y$ must satisfy the equations of both lines (as $A$ is on both lines). Substituting the first equation into the second gives: \\begin{align*}\n2x+5y &=11\\\\\n6y+5y&= 11\\\\\n11y&= 11\\\\\ny &= 1.\n\\end{align*}So now $x = 3y = 3$, and so the coordinates of point $A$ are $(3,1)$. The sum of these is 3+1 = #### 4"
            }
            cot_ex['example5'] = {
                "Question":"Example Problem: What is the positive value of the expression $\\sqrt{x^3 - 2^y}$ when $x = 5$ and $y = 2$?",
                "Answer": "Plugging in, the desired expression is just $\\sqrt{5^3 - 2^2} = \\sqrt{125 - 4} = \\sqrt{121} =11$. The answer is #### 11"
            }
            cot_ex['example6'] = {
                "Question": "Example Problem: A Senate committee has 5 Democrats and 5 Republicans.  In how many distinguishable ways can they sit around a circular table if all the members of each party all sit next to each other?  (If the table is rotated, it should be counted as the same seating.)",
                "Answer": "Choose any 5 consecutive seats in which to place the Democrats -- it doesn't matter which 5 consecutive seats that we choose, since we can rotate the table.  Then there are $5!$ ways to place the Democrats in their seats, and $5!$ ways to place the Republicans in their seats, for a total of $5! \\times 5! = 14,\\!400$ arrangements. The answer is #### 14,\\!400"
            }
            cot_ex['example7'] = {
                "Question":"Example Problem: Evaluate $\\frac{7!+8!}{6!+7!}$. Express your answer as a common fraction.",
                "Answer": "We notice that 6! is a common factor of the numerator and denominator and then simplify: \\begin{align*}\n$\\frac{7!+8!}{6!+7!} &= \\frac{7\\cdot6!+8\\cdot7\\cdot6!}{6!+7\\cdot6!} \\\\\n&=\\frac{6!(7+8\\cdot7)}{6!(1+7)} \\\\\n&=\\frac{7+56}{1+7}\\\\\n&=\\boxed{\\frac{63}{8}}$\\\\\n\\end{align*} The answer is #### \\frac{63}{8}"
            }
            cot_ex['example8'] = {
                "Question":"Example Problem: In how many ways can the letters of the word BANANA be rearranged such that the new word does not begin with a B?",
                "Answer": "The first thing to do is place the B since it has a restriction on it. We can put it anywhere but the first place, so we have 5 options. Once we have done that, we just need to place the two N's and then the rest of the spots will just be A's. We have 5 spots left, so there are 5 options for where to put the first N and 4 options for where to put the second N. However, the two N's are identical, which means that we've counted each new word twice. So our answer is $\\frac{5\\times5\\times4}{2}=50$. The answer is #### 50"
            }
        elif dataset in ["triviaqa","nq"]:
            cot_ex['example1'] = {
                "Question": "Which Lloyd Webber musical premiered in the US on 10th December 1993?",
                "Answer": "To determine which Lloyd Webber musical premiered in the US on December 10, 1993, we start by identifying the musicals composed by Andrew Lloyd Webber. His well-known works include \"The Phantom of the Opera,\" \"Cats,\" \"Evita,\" \"Jesus Christ Superstar,\" \"Joseph and the Amazing Technicolor Dreamcoat,\" and \"Sunset Boulevard.\" By narrowing down based on the premiere date, we find that \"Sunset Boulevard\" is the musical that matches this specific date. Verifying historical records confirms that \"Sunset Boulevard\" indeed premiered in the US on December 10, 1993. Therefore, the Lloyd Webber musical that premiered in the US on December 10, 1993, is #### Sunset Boulevard"
            }
            cot_ex['example2'] = {
                "Question": "Who was the target of the failed \"Bomb Plot\" of 1944?",
                "Answer": "The target of the failed \"Bomb Plot\" of 1944 was Adolf Hitler. This assassination attempt, also known as the July 20 Plot, was carried out by German military officers and civilians who were part of the German resistance to overthrow the Nazi regime. The plot involved placing a bomb in a briefcase under a table in a conference room at Hitler's Wolf's Lair headquarters in East Prussia. Although the bomb exploded, Hitler survived the assassination attempt. #### Adolf Hitler"
            }
            cot_ex['example3'] = {
                "Question": "Who directed the classic 30s western Stagecoach?",
                "Answer": "The classic 30s western \"Stagecoach\" was directed by #### John Ford"
            }
            cot_ex['example4'] = {
                "Question": "What was President Gerald Ford's middle name?",
                "Answer": "President Gerald Ford's middle name was #### Rudolph"
            }
            cot_ex['example5'] = {
                "Question": "Art Garfunkel trained for which profession although he didn't qualify?",
                "Answer": "Art Garfunkel trained for the profession of mathematics teacher although he didn't qualify. #### mathematics teacher"
            }
            cot_ex['example6'] = {
                "Question": "Kagoshima international airport is in which country?",
                "Answer": "Kagoshima International Airport is in #### Japan"
            }
            cot_ex['example7'] = {
                "Question": "Who had a Too Legit To Quit Tour?",
                "Answer": "The artist who had a Too Legit To Quit Tour was #### MC Hammer"
            }
            cot_ex['example8'] = {
                "Question": "In which American state is the Isabella Stewart Gardner Museum?",
                "Answer": "The Isabella Stewart Gardner Museum is in the American state of #### Massachusetts"
            }
    else:
        if dataset == "mmlu":
            cot_ex['example1'] = "What is the value of p in 24 = 2p?\nOptions: A:p=4, B:p=8, C:p=12, D:p=24\nAnswer: To find the value of  p  in the equation  24 = 2p , we need to solve for  p  step by step. First, we start with the given equation, which is  24 = 2p . To isolate  p , we divide both sides of the equation by 2. This gives us  24 \div 2 = 2p \div 2 . Simplifying both sides, we get  12 = p . Therefore, the value of  p  is 12. Thus, the correct answer is option #### C"
            cot_ex['example2'] = "Ms. Perez drove a total of 40 miles in 5 days. She drove the same number of miles each day. How many miles did Ms. Perez drive each day? \Options: A:5, B:7, C:8, D:9\nAnswer: Ms. Perez drove a total of 40 miles over the course of 5 days. Since she drove the same number of miles each day, we can determine how many miles she drove each day by dividing the total miles by the number of days. Dividing 40 miles by 5 days gives us 8 miles per day. Therefore, the answer is option #### C"
            cot_ex['example3'] = "What is the remainder of 21 divided by 7?\nOptions: A:21, B:7, C:1, D:None of these\nAnswer: To find the remainder when 21 is divided by 7, we divide 21 by 7. The result of the division is 3, which is an integer, meaning there is no remainder. Thus, when 21 is divided by 7, it divides evenly with no remainder left over. Therefore, the remainder of 21 divided by 7 is 0. The answer is #### D"
            cot_ex['example4'] = "Identify the algebraic expression that matches the word phrase: 4 times as many cans as Tom collected.\nOptions: A:4t, B:t over 4, C:t - 4, D:t + 4\nAnswer: To identify the algebraic expression that matches the phrase “4 times as many cans as Tom collected,” we start by letting  x  represent the number of cans Tom collected. The phrase “4 times as many” means that we need to multiply the number of cans Tom collected by 4. Therefore, the algebraic expression that represents “4 times as many cans as Tom collected” is  4x. The answer is #### A"
            cot_ex['example5'] = "What is the value of the expression 2(3(4^2 + 1)) - 2^3?\nOptions: A:156, B:110, C:94, D:48\nAnswer: To find the value of the expression 2(3(4^2 + 1)) - 2^3, we need to evaluate it step by step in a detailed manner: First, we calculate the exponent 4^2: 4^2 = 16 Next, we add 1 to this result: 16 + 1 = 17 Then, we multiply this result by 3: 3 \times 17 = 51 After that, we multiply the result by 2: 2 \times 51 = 102 Next, we evaluate the exponent 2^3: 2^3 = 8 Finally, we subtract this result from 102: 102 - 8 = 94 Thus, the value of the expression 2(3(4^2 + 1)) - 2^3 is 94. The answer is option #### C"
            cot_ex['example6'] = "Compute 22 / 2 + 9.\nOptions: A:10, B:11, C:20, D:2\nAnswer: To compute the expression 22 / 2 + 9, we start by performing the division first. Dividing 22 by 2 gives us 11. Next, we add 9 to the result of the division. Therefore, 11 plus 9 equals 20. Thus, the value of the expression 22 / 2 + 9 is 20. The answer is #### C"
            cot_ex['example7'] = "What is the least common multiple of 4 and 10?\nOptions: A:14, B:20, C:40, D:60\nAnswer: To find the least common multiple (LCM) of 4 and 10, we follow these steps in a detailed manner: First, we determine the prime factors of each number. The prime factors of 4 are 2 and 2, which means 4 = 2 \times 2 = 2^2. The prime factors of 10 are 2 and 5, which means 10 = 2 \times 5. Next, we identify the highest power of each prime factor that appears in either number. For the prime number 2, the highest power that appears is 2^2 (from the number 4). For the prime number 5, the highest power that appears is 5 (from the number 10). We then multiply these highest powers together to find the LCM. We multiply 2^2 and 5 together: 2^2 \times 5 = 4 \times 5 = 20. Therefore, the least common multiple of 4 and 10 is 20. The answer is #### B"
            cot_ex['example8'] = "Estimate 711 + 497. The sum is between which numbers?\nOptions: A:50 and 400, B:450 and 700, C:750 and 1000, D:1,050 and 1,300\nAnswer:To estimate the sum of 711 and 497, let’s round each number to the nearest hundred and then add them together. 711 rounds to 700 and 497 rounds to 500. Now, add the rounded numbers: 700 + 500 = 1200. The sum of 711 and 497 is close to 1200, which lies between 1050 and 1300. Therefore, the answer is #### D"
        elif dataset == "mmlu_short":
            cot_ex['example1'] = "What is the value of p in 24 = 2p?\nAnswer: To find the value of  p  in the equation  24 = 2p , we need to solve for  p  step by step. First, we start with the given equation, which is  24 = 2p . To isolate  p , we divide both sides of the equation by 2. This gives us  24 \div 2 = 2p \div 2 . Simplifying both sides, we get  12 = p . Therefore, the value of  p  is 12. Thus, the correct answer is #### p=12"
            cot_ex['example2'] = "Ms. Perez drove a total of 40 miles in 5 days. She drove the same number of miles each day. How many miles did Ms. Perez drive each day?\nAnswer: Divide the total miles by the number of days: 40 ÷ 5 = 8. Therefore, the answer is #### 8."
            cot_ex['example3'] = "What is the remainder of 21 divided by 7?\nAnswer: 21 ÷ 7 = 3 with no remainder. Therefore, the remainder is 0. The answer is #### 0."
            cot_ex['example4'] = "Identify the algebraic expression that matches the word phrase: 4 times as many cans as Tom collected.\nAnswer: Let x represent the number of cans Tom collected. '4 times as many' means 4x. Therefore, the answer is #### 4x."
            cot_ex['example5'] = "What is the value of the expression 2(3(4^2 + 1)) - 2^3?\nAnswer: Calculate step by step: 4^2 = 16, 16 + 1 = 17, 3 × 17 = 51, 2 × 51 = 102, 2^3 = 8, 102 - 8 = 94. Thus, the value is 94. The answer is #### 94."
            cot_ex['example6'] = "Compute 22 / 2 + 9.\nAnswer: 22 ÷ 2 = 11, 11 + 9 = 20. Therefore, the answer is #### 20."
            cot_ex['example7'] = "What is the least common multiple of 4 and 10?\nAnswer: Prime factors of 4 are 2^2 and of 10 are 2 and 5. LCM is 2^2 × 5 = 4 × 5 = 20. Therefore, the answer is #### 20."
            cot_ex['example8'] = "Estimate 711 + 497. The sum is between which numbers?\nAnswer: Round 711 to 700 and 497 to 500. 700 + 500 = 1200. Therefore, the sum is between 1050 and 1300. The answer is #### 1200."
        
        elif dataset == "aqua":
            cot_ex['example1'] = "A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower?\nOptions: A)5(√3 + 1), B)6(√3 + √2), C)7(√3 – 1), D)8(√3 – 2), E)None of these\nAnswer : Let the height of the building be h. Initially, he was at an angle of 450. tan 45 = h/distance between car and tower. h = distance between car and tower (since tan 45 = 1). Now, after 10 minutes, it travelled a certain distance, and angle changed to 600. tan 60 = h/x x = h/√3 So, in 10 minutes, it has travelled a distance of h – x = h - h/√3. 10 minutes = h *( 1 – 1√3) h can be travelled in 10 / (1 – 1√3). To travel a distance of x, which is h/√3, it takes : h = 10 / (1 – 1/√3) h / √3 = 10/ √3 * (1 – 1/√3). Multiply numerator and denominator by 1 + √3 ( conjugate of 1 - √3). We get, x = h/√3 = 10 (1 + √3) / 2 = 5* (1 + √3) So, it takes 5(1 + √3) minutes to reach the base of the tower. Answer: #### A"
            cot_ex['example2'] = "Find out which of the following values is the multiple of X, if it is divisible by 9 and 12?\nOptions: A)36, B)15, C)17, D)5, E)7\nAnswer:9=3*3 12=3*4 The number should definitely have these factors 3*3*4 36 is the number that has these factors So, 36 is the multiple of X Answer is #### A"
            cot_ex['example3'] = "A trader sold an article at a profit of 20% for Rs.360. What is the cost price of the article?\nOptions: A)270, B)300, C)280, D)320, E)315\nAnswer:Cost Price = Selling Price / (100+Profit%) × 100 => 360 / (100+20) × 100 => 360 / 120 × 100 = Rs.300 Option #### B"
            cot_ex['example4'] = "Find the total no. of distinct bike no.'s that can beformed using 2 letters followed by 2 no.'s. How many letters need to be distinct?\nOptions: A)74453, B)64543, C)74325, D)65000, E)97656\nAnswer: Out of 26 alphabets two distinct letters can be chosen in 26P2 ways. Coming to the numbers part, there are 10 ways to choose the first digit and similarly, there are another 10 ways to choose the second digit. Hence, there are in total 10X10 = 100 ways. Combined with letters there are 6P2 X 100 ways = 65000 ways to choose vehicle numbers. #### D"
            cot_ex['example5'] = "[5 + ? × 19 - 15 - 7]/[13 × 13 - 156] = 6\nOptions: A)4, B)4.5, C)5, D)5.5, E)6.5\nAnswer: (? × 19 - 17)/(169 - 156) = 6 => ? × 19 - 17 = 13 × 6 = 76 => ? × 19 = 78 + 17 = 95 ? = 95/19 = 5 Answer: Option #### C"
            cot_ex['example6'] = "A book was sold for Rs 27.50 with a profit of 10%. If it were sold for Rs. 25.75, then would have been percentage of profit and loss ?\nOptions: A)2% Profit, B)3% Profit, C)2% Loss, D)3% Loss, E)4% Loss\nAnswer: S.P.=(100+gain%100∗C.P) So, C.P. = (100/110∗25.75) When S.P. = 25.75 then Profit=25.75−25=Re.0.75 Profit%=0.75/25∗100=3% Answer is #### B"
            cot_ex['example7'] = "900 + 5 × 12 = ?\nOptions: A)820, B)202, C)420, D)209, E)960\nAnswer: 900 + 5 × 12 = ? or, ? = 900 + 60 = 960 Answer #### E"
            cot_ex['example8'] = "What is the units digit of 9^3-7?\nOptions: A)1, B)3, C)5, D)2, E)4\nAnswer: The unit's digit of 9^3 = 9 9-7=2 Answer #### D"
        elif dataset == "race":
            cot_ex['example1'] = """I am a psychologist. I first met Timothy, a quiet, overweight eleven-year-old boy, when his mother brought him to me to discuss his declining grades. A few minutes with Timothy were enough to confirm that his self-esteem and general happiness were falling right along with _ . I asked about Timothy's typical day. He awoke every morning at six thirty so he could reach his school by eight and arrived home around four thirty each afternoon. He then had a quick snack, followed by either a piano lesson or a lesson with his math tutor. He finished dinner at 7 pm, and then he sat down to do homework for two to three hours. Quickly doing the math in my head, I found that Timothy spent an average of thirteen hours a day at a writing desk. What if Timothy spent thirteen hours a day at a sewing machine instead of a desk? We would immediately be shocked, because that would be called children being horribly mistreated. Timothy was far from being mistreated, but the mountain of homework he faced daily resulted in a similar consequence --he was being robbed of his childhood. In fact, Timothy had no time to do anything he truly enjoyed, such as playing video games, watching movies, or playing board games with his friends. Play, however, is a crucial part of healthy child development. It affects children's creativity, their social skills, and even their brain development. The absence of play, physical exercise, and freefrom social interaction takes a serious toll on many children. It can also cause significant health problems like childhood obesity, sleep problems and depression. Experts in the field recommend the minutes children spend on their homework should be no more than ten times the number of their grade level. As a fifthgrader, Timothy should have no more than fifty minutes a day of homework (instead of three times that amount). Having an extra two hours an evening to play, relax, or see a friend would soundly benefit any child's life quality.\nQuestion: What did the writer think of Timothy after learning about his typical day?\nOptions: A: "Timothy was very hardworking.", B: "Timothy was being mistreated.", C: "Timothy had a heavy burden.", D: "Timothy was enjoying his childhood." \nAnswer: Observation: Timothy's grades, self-esteem, and happiness were declining. Daily Routine: Timothy spends around thirteen hours a day working at a desk with school, lessons, and homework. Writer’s Comparison: The writer compares this to child mistreatment, highlighting the excessive burden. Health and Development: Lack of playtime affects creativity, social skills, brain development, and leads to health issues like obesity and depression. Expert Advice: Homework should be limited to fifty minutes for a fifth-grader, not the excessive amount Timothy has. The writer thinks Timothy has a heavy burden. So, the correct answer is: #### C"""
            cot_ex['example2'] = """From self-driving cars to carebots (care+robots) for elderly people, rapid development in technology has long represented a possible threat to many jobs normally performed by people. But experts now believe that almost 50 percent of occupations existing today will be completely unnecessary by 2025 as artificial intelligence continues to change businesses. "The next fifteen years will see a revolution in how we work, and a revolution will necessarily take place in how we plan and think about workplaces," said Peter Andrew, Director of Workplace Strategy for CBRE Asia Pacific. A growing number of jobs in the future will require creative intelligence, social skills and the ability to use artificial intelligence. The report is based on interviews with 200 experts, business leaders and young people from Asia Pacific, Europe and North America. It shows that in the US technology already destroys more jobs than it creates. But the report states: "Losing occupations does not necessarily mean losing jobs -- just changing what people do." Growth in new jobs could occur as much, according to the research. "The growth of 20 to 40 person companies that have the speed and technological know-how will directly challenge big companies," it states. A 2014 report by Pew Research found 52 percent of experts in artificial intelligence and robotics were optimistic about the future and believed there would still be enough jobs in the next few years. The optimists pictured "a future in which robots do not take the place of more jobs than they create," according to Aaron Smith, the report's co-author. "Technology will continue to affect jobs, but more jobs seem likely to be created. Although there have always been unemployed people, when we reached a few billion people there were billions of jobs. There is no shortage of things that need to be done and that will not change," Microsoft's Jonathan Grudin told researchers.\nQuestion: According to the report, _ . \nOptions:  A: "people won't necessarily lose jobs", B: "big companies will face fewer challenges", C: "small companies will win against big companies", D: "most people will become interested in technology" \nAnswer: Let's break down the information provided to determine the correct answer. Technology is rapidly developing and poses a threat to many jobs traditionally performed by people. Experts believe that almost 50% of current occupations will be unnecessary by 2025 due to artificial intelligence. The future will demand jobs requiring creative intelligence, social skills, and the ability to use AI. Technology is already destroying more jobs than it creates in the US. However, losing occupations doesn't necessarily mean losing jobs, but rather changing what people do. Growth is expected in small, agile companies with technological expertise. 52% of experts believe there will still be enough jobs in the future. There is a belief that robots will not replace more jobs than they create. Despite technological impacts, there will always be things that need to be done, hence job creation will continue.From this breakdown, the statement that matches the report's findings is: A: "people won't necessarily lose jobs. So, the correct answer is: #### A"""
            cot_ex['example3'] = """If you want to be a volunteer, you have to answer the following typical questions. Do you want to work with people, animals or machines? Do you want to work indoors or outdoors, directly serve people in need or serve people behind the scenes? Every year, thousands of people in the west offer volunteer service. Volunteering greatly strengthens the community because it helps the old, the young, the weak, the sick, and the disabled and the injured to solve problems. Volunteers usually help in many different ways. They may give people advice, offer friendship to the young, drive the elderly to church (if up to the driving age), advise kids against drugs, work as assistants in schools or nursing homes, raise funds, plant trees, help out in local libraries and do many other things. Volunteering can be a few hours a week or a few hours a month. Anybody who wants to serve people in need can become a volunteer. In fact, the art of volunteering is a process of both giving and receiving. Volunteering allows volunteers to meet new people, make new friends and mix with people from all walks of life. Volunteering is an excellent way to experiment and try out new techniques and skills, discover your individual talents and explore career choice. Being a volunteer will take you on a wonderful journey and help you learn more than what you can get from books.\nQuestion: What's the best title of the passage?\nOptions: A: "How to hunt for jobs", B: "Volunteer service in the west", C: "How to make friends", D: "How to work with animals" \nAnswer: The passage discusses various aspects of volunteering, including the types of volunteer work, the benefits to the community, and the personal benefits to the volunteers. The focus is on volunteer service in the West and its impact. Given this information, the best title for the passage is: B: "Volunteer service in the west" So, the correct answer is: #### B"""
            cot_ex['example4'] = """Let us take a serious look at what the results might be if such a suggestion were accepted. Families might use the time for a real family hour. They might sit around together after dinner and actually talk to one another. It is well known that many of our problems-everything, in fact, from the generation gap to the high divorce rate to some forms of mental illnesses-are caused at least in part by the failure to communicate. We do not tell each other what is disturbing us. The result is emotional difficulty of one kind or another. By using the quiet family hour to discuss our problems, we might get to know each other better, and to like each other better. On evenings when such talk is unnecessary, families could rediscover more active entertainments. Freed from TV, forced to find their own activities, they might take a ride together to watch the sunset. Or they might take a walk together and see the neighborhood with fresh, new eyes. With free time and no TV, children and adults might rediscover reading. There is more entertainment in a good book than in a month of typical TV programming. Educators report that the generation growing up with television can hardly write an English sentence, even at the college level. Writing is often learned from reading. A more cultured new generation could be a product of the quiet hour. At first glance, the idea of an hour without TV seems radical . What will parents do without the electronic baby-sitter? How will we spend the time? But it's not radical at all. It has been only twenty-five years since television came to control our free time. Those of us thirty-five and older can remember childhoods without television, spent partly with radio-which at least involved the listener's imagination-but also with reading, learning, talking, playing games and inventing new activities. It wasn't that difficult, honest. The truth is that we had a ball.\nQuestion: According to the text, good books _ .\nOptions: A: "offer as much knowledge as TV does", B: "are more entertaining than TV", C: "should be made into films", D: "cannot be found easily"\nAnswer: According to the text, good books are more entertaining than TV. So, the correct answer is: B: "are more entertaining than TV. The answer is #### B"""
        elif dataset == "gsm8k":
            cot_ex['example1'] = "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer: We start with 15 trees. Later we have 21 trees.The difference must be the number of trees they planted. So, they must have planted 21-15=6 trees.The answer is #### 6."
            cot_ex['example2'] = "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is #### 5."
            cot_ex['example3'] = "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is #### 39."
            cot_ex['example4'] = "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is #### 8."
            cot_ex['example5'] = "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is #### 9."
            cot_ex['example6'] = "There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?\nAnswer: There are 4 days from Monday to Thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is #### 29."
            cot_ex['example7'] = "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?\nAnswer: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is #### 33."
            cot_ex['example8'] = "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is #### 8."
        elif dataset == "math":
            cot_ex['example1'] = "Evaluate $\\lfloor 14.6 \\rfloor-\\lceil-14.6\\rceil$.\nAnswer: The greatest integer less than $14.6$ is $14$. The smallest integer greater than $-14.6$ is $-14$. Therefore, the equation can be rewritten as $14-(-14)$, or #### 28"
            cot_ex['example2'] = "What is the sum of all numbers $a$ for which the graph of $y=x^2+a$ and the graph of $y=ax$ intersect one time?\nAnswer: If these two graphs intersect then the point of intersection occur when  \\[x^2+a=ax,\\] or  \\[x^2-ax+a=0.\\] This quadratic has one solution exactly when the discriminant is equal to zero: \\[(-a)^2-4\\cdot1\\cdot a=0.\\] This simplifies to  \\[a(a-4)=0.\\]\n\nThere are exactly two values of $a$ for which the line and parabola intersect one time, namely $a=0$ and $a=4$.  The sum of these values is  #### 4"
            cot_ex['example3'] = "At 50 miles per hour, how far would a car travel in $2\\frac{3}{4}$ hours? Express your answer as a mixed number.\nAnswer: In two hours, a car traveling at $50$ mph would travel $50$ mph $\\times 2$ hours $= 100$ miles. Now we find how far a car can travel in $3/4$ of an hour which is $50$ mph $\\times \\frac{3}{4}$ hours $ = \\frac{150}{4} = 37 \\frac{1}{2}$ miles. Thus, the car would travel a total of $100 + 37 \\frac{1}{2}= 137 \\frac{1}{2}$ miles. The answer is ####  137 \\frac{1}{2}"
            cot_ex['example4'] = "In a rectangular coordinate system, the line $3y = x$ intersects the line $2x + 5y = 11$ at point $A$. What is the sum of the coordinates of point $A$?\nAnswer: If the coordinates of point $A$ are $(x,y)$ then $x$ and $y$ must satisfy the equations of both lines (as $A$ is on both lines). Substituting the first equation into the second gives: \\begin{align*}\n2x+5y &=11\\\\\n6y+5y&= 11\\\\\n11y&= 11\\\\\ny &= 1.\n\\end{align*}So now $x = 3y = 3$, and so the coordinates of point $A$ are $(3,1)$. The sum of these is $3+1 = #### 4"
            cot_ex['example5'] = "What is the positive value of the expression $\\sqrt{x^3 - 2^y}$ when $x = 5$ and $y = 2$?\nAnswer: Plugging in, the desired expression is just $\\sqrt{5^3 - 2^2} = \\sqrt{125 - 4} = \\sqrt{121} = #### 11$."
            cot_ex['example6'] = "A Senate committee has 5 Democrats and 5 Republicans.  In how many distinguishable ways can they sit around a circular table if all the members of each party all sit next to each other?  (If the table is rotated, it should be counted as the same seating.)\nAnswer: Choose any 5 consecutive seats in which to place the Democrats -- it doesn't matter which 5 consecutive seats that we choose, since we can rotate the table.  Then there are $5!$ ways to place the Democrats in their seats, and $5!$ ways to place the Republicans in their seats, for a total of $5! \\times 5! = 14,\\!400$ arrangements. The answer is #### 14,\\!400"
            cot_ex['example7'] = "Evaluate $\\frac{7!+8!}{6!+7!}$. Express your answer as a common fraction.\nAnswer: We notice that 6! is a common factor of the numerator and denominator and then simplify: \\begin{align*}\n$\\frac{7!+8!}{6!+7!} &= \\frac{7\\cdot6!+8\\cdot7\\cdot6!}{6!+7\\cdot6!} \\\\\n&=\\frac{6!(7+8\\cdot7)}{6!(1+7)} \\\\\n&=\\frac{7+56}{1+7}\\\\\n&=\\boxed{\\frac{63}{8}}$\\\\\n\\end{align*} The answer is #### \\frac{63}{8}"
            cot_ex['example8'] = "In how many ways can the letters of the word BANANA be rearranged such that the new word does not begin with a B?\nAnswer: The first thing to do is place the B since it has a restriction on it. We can put it anywhere but the first place, so we have 5 options. Once we have done that, we just need to place the two N's and then the rest of the spots will just be A's. We have 5 spots left, so there are 5 options for where to put the first N and 4 options for where to put the second N. However, the two N's are identical, which means that we've counted each new word twice. So our answer is $\\frac{5\\times5\\times4}{2}=50$. The answer is #### 50"
        elif dataset in ["triviaqa", "nq"]:
            cot_ex['example1'] = """Which Lloyd Webber musical premiered in the US on 10th December 1993?\nAnswer: To determine which Lloyd Webber musical premiered in the US on December 10, 1993, we start by identifying the musicals composed by Andrew Lloyd Webber. His well-known works include "The Phantom of the Opera," "Cats," "Evita," "Jesus Christ Superstar," "Joseph and the Amazing Technicolor Dreamcoat," and "Sunset Boulevard." By narrowing down based on the premiere date, we find that "Sunset Boulevard" is the musical that matches this specific date. Verifying historical records confirms that "Sunset Boulevard" indeed premiered in the US on December 10, 1993. Therefore, the Lloyd Webber musical that premiered in the US on December 10, 1993, is #### Sunset Boulevard"""
            cot_ex['example2'] = """Who was the target of the failed ""Bomb Plot"" of 1944?\nAnswer: The target of the failed "Bomb Plot" of 1944 was Adolf Hitler. This assassination attempt, also known as the July 20 Plot, was carried out by German military officers and civilians who were part of the German resistance to overthrow the Nazi regime. The plot involved placing a bomb in a briefcase under a table in a conference room at Hitler's Wolf's Lair headquarters in East Prussia. Although the bomb exploded, Hitler survived the assassination attempt. #### Adolf Hitler"""
            cot_ex['example3'] = """Who directed the classic 30s western Stagecoach?\nAnswer: The classic 30s western "Stagecoach" was directed by #### John Ford"""
            cot_ex['example4'] = """What was President Gerald Ford's middle name?\nAnswer: President Gerald Ford's middle name was #### Rudolph"""
            cot_ex['example5'] = """Art Garfunkel trained for which profession although he didn't qualify?\nAnswer: Art Garfunkel trained for the profession of mathematics teacher although he didn't qualify. #### mathematics teacher"""
            cot_ex['example6'] = """Kagoshima international airport is in which country?\nAnswer: Kagoshima International Airport is in #### Japan"""
            cot_ex['example7'] = """Who had a Too Legit To Quit Tour?\nAnswer: The artist who had a Too Legit To Quit Tour was #### MC Hammer"""
            cot_ex['example8'] = """In which American state is the Isabella Stewart Gardner Museum?\nAnswer: The Isabella Stewart Gardner Museum is in the American state of #### Massachusetts"""
        
    return cot_ex
def get_model(model, cache_dir, hf_token):
    if model == "llama3_8b":
        model = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", download_dir=cache_dir)
    elif model == "gemma_7b":
        model = LLM(model="google/gemma-1.1-7b-it", download_dir=cache_dir)
    elif model == "mistral_7b":
        model = LLM(model="mistralai/Mistral-7B-Instruct-v0.1", download_dir=cache_dir)
    elif model == "qwen2_7b":
        model = LLM(model="Qwen/Qwen2-7B-Instruct", download_dir=cache_dir)
    elif model == "llama3_70b":
        model = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct", download_dir =cache_dir, tensor_parallel_size=4, dtype="bfloat16")
    return model
def main():
    hf_token = "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"
    cache_model = "../../hub/model/"
    cache_data = "../../data/"
    #model = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", download_dir=cache_model)
    model_list = ["llama3_8b", "gemma_7b", "mistral_7b", "qwen2_7b", "llama3_70b"]
    selected_model = args.model
    selected_dataset = args.dataset  #### mmlu  race  aqua  gsm8k  math  arc  
    model = get_model(selected_model, cache_model, hf_token)
    print(f"Model: {selected_model} is selected")
    dataset_config = {"mmlu":["question","answer","choices",None],
                      "mmlu_short":["question","answer","choices",None],
                      "race":["question","answer","options","article"],
                      "aqua":["question","correct","options",None],
                      "gsm8k":["question","answer",None,None],
                      "math":["problem","solution",None,None],
                      "triviaqa":["question","answer",None,None],
                      "nq":["question", "answer", None, None]
                      }
    if selected_dataset in ["mmlu", "mmlu_short"]:
        dataset = load_dataset("cais/mmlu",'all', cache_dir=cache_data)
        final_df = dataset['test'].shuffle(seed=42)
    elif selected_dataset == "race":
        dataset = load_dataset("ehovy/race",'all', cache_dir=cache_data)
        final_df = dataset['train'].shuffle(seed=42)
    elif selected_dataset == "aqua":
        dataset = load_dataset("deepmind/aqua_rat", cache_dir=cache_data)
        final_df = dataset['train'].shuffle(seed=42)
    elif selected_dataset == "gsm8k":
        dataset = load_dataset("gsm8k", 'main', cache_dir=cache_data)
        final_df = dataset['test']
    elif selected_dataset == "math":
        dataset = load_dataset("hendrycks/competition_math", trust_remote_code=True, cache_dir=cache_data)
        final_df = dataset['train'].shuffle(seed=42)
    elif selected_dataset == "triviaqa":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", cache_dir=cache_data)
        final_df = dataset['train'].shuffle(seed=42)
    elif selected_dataset == "nq":
        dataset = load_dataset("google-research-datasets/nq_open", cache_dir=cache_data)
        final_df = dataset['train'].shuffle(seed=42)
    print(f"Dataset: {selected_dataset} is selected")
    q_col, a_col, o_col, c_col = dataset_config[selected_dataset]

    # 데이터셋 로드
    q_dataset = final_df[q_col][:1000]
    a_dataset = final_df[a_col][:1000]
    if o_col is not None:
        o_dataset = final_df[o_col][:1000]
    else:
        o_dataset = None
    if c_col is not None:
        c_dataset = final_df[c_col][:1000]
    else:
        c_dataset = None
    results = []
    cot_ex = get_cot_Ex(selected_dataset, selected_model)
    correct = 0
    total = 0 
    results = parallel_process(context=c_dataset, question=q_dataset, answer=a_dataset, option=o_dataset, cot_ex=cot_ex, model=model, model_name=selected_model, dataset=selected_dataset)

    for result in results:
        question, real_answer, original_most_common, original_is_correct, original_conf, original_hr, original_rp = result
        if original_is_correct:
            correct += 1
        total += 1       
    accuracy = correct / total
    print(f'Total Accuracy: {accuracy}')
    df = pd.DataFrame(results, columns=['question', 'real_answer', 'original_most_common', 'original_is_correct', 'original_conf', 'original_hr', 'original_rp'])
    df.to_pickle(f'inter_SC_beam_expand_{selected_model}_{selected_dataset}_{args.n_init}_{args.n_filter}_{args.n_output_per_one}_interval_{args.n_tokens_per_iter}.pkl')
    #print("Pkl 파일로 저장 완료")
if __name__ == "__main__":
    #import os
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()