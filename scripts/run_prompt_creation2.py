import os
from datasets import load_dataset, load_from_disk
import pandas as pd
from tqdm import tqdm
import argparse

prompt = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The speaker's name
2. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
4. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
5. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)
6. The pitch of the speaker's voice (very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch)

Your task is to create a text description using these keywords that accurately describes the speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description. 
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description. 
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

For example, given the following keywords: 'Yusuf', 'slightly distant-sounding', 'noisy', 'very expressive and animated', 'very slowly', 'moderate pitch', a valid description would be: 'Yusuf speaks very slowly but has a very animated delivery. The recording is noisy and there is some roominess.'
Another valid description would be: 'In a noisy room, Yusuf delivers a very animated and expressive speech, at a very slow pace.'
Another valid description would be: 'Yusuf enunciates a very expressive speech. Yusuf's voice is slightly distant-sounding, with some background noise present. Yusuf speaks very slowly with a moderate pitch but a very expressive tone.'

DO NOT use gendered pronouns. Keep the speaker's gender unknown/neutral in the description

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: '[speaker]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]', the corresponding description is:
"""

def prepare_prompt(example):
    speaker = example['speaker']
    reverb = example['reverberation']
    noise = example['sdr_noise']
    monotony = example['speech_monotony']
    rate = example['speaking_rate']
    pitch = example['pitch']
    prompt = prompt.replace('[speaker]', speaker).replace('[reverberation]', reverb).replace('[sdr_noise]', noise).replace('[speech_monotony]', monotony).replace('[speaking_rate]', rate).replace('[pitch]', pitch)
    example['prompt'] = prompt
    return example

def _send(data, url):
    NON_RETRYABLE_CODES = {400, 401, 403, 404, 405, 409, 413, 422}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in NON_RETRYABLE_CODES:
            E = e.response.status_code
            logger.error(f"Non-retryable error: {E}")
            print(str(e))
            return None
        logger.error(f"HTTP error occurred: {e}")
        raise
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Request failed: {e.response.status_code}")
        else:
            logger.error(f"Request failed: {str(e)}")
        raise

def infer(model, messages):
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 5000,
        "temperature": 0,
        "top_p": 1,
    }

    response_json = _send(data, "localhost:8000")
    if response_json is None:
        return "<placeholder>"

    text = response_json['choices'][0]['message']['content'].strip()
    return text

def main(ds_name, model, test):
    ds_name_short = ds_name.split("/")[-1] 
    ds_name_pq = ds_name_short + "_cached.parquet"
    try:
        ds = load_from_disk(ds_name_short)
        print("Loaded dataset from disk")
    except Exception as e:
        print(str(e))
        ds = load_dataset(ds_name)
    try:
        ds = ds['train']
    except:
        pass
    if test:
        ds = ds.select(range(100))
        
    try:
        prompts = pd.read_parquet(ds_name_pq)["prompt"]
        print("Loaded cached prompts")
    except:
        print("Cached prompts not found. Recreating...")
        ds = ds.map(prepare_prompt, num_proc=8)
        prompts = list(ds["prompt"])
        pd.DataFrame({"prompt":prompts}).to_parquet(ds_name_pq, index=False)
        
    annotations = []
    for prompt in tqdm(prompts, desc="Annotating..."):
        messages = [{"role": "user", "messages": prompt}]
        output = infer(model, messages)
        annotations.append(output)
    fails = annotations.count("<placeholder>")
    print(f"Failed annotations: {fails}")
    ds = ds.add_column("annotation", annotations)
    ds.save_to_disk(ds_name_short + "+annotated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, required=True, help='dataset name')
    parser.add_argument('--model', type=str, required=True, help='vllm model name')
    parser.add_argument('--test', action='store_true', default=False, help='test with 100 samples')
    args = parser.parse_args()
    ds_name, model, test = args.ds_name, args.model, args.test
    main(ds_name, model, test)
