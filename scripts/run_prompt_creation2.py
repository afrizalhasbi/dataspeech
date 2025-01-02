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

def main(ds_name):
    ds_name_pq = ds_name.split("/")[1] + "_cached.parquet"
    try:
        ds = load_from_disk(ds_name_short)
    except:
        ds = load_dataset(ds_name)['train']
        
    try:
        prompts = pd.read_parquet(ds_name_pq)["prompt"]
        print("Loaded cached prompts")
    except:
        print("Cached prompts not found. Recreating...")
        ds = ds.map(prepare_prompt, num_proc=8)
        prompts = list(ds["prompt"])
        pd.DataFrame({"prompt":prompts}).to_parquet(ds_name_pq, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, required=True, help='dataset name')
    main(ds_name)
