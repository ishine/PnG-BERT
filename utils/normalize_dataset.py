from this import d
import numpy as np
import os
import string
import tqdm
from nemo_text_processing.text_normalization.normalize import Normalizer
from concurrent.futures import ProcessPoolExecutor

def normalize_parallel(normalizer, text_list):
    normalized = normalizer.normalize_list(text_list)
    return normalized

def main(src_text_file, dest_txt_file):
    normalizer = Normalizer(input_case='lower_cased', lang='en')
    executor = ProcessPoolExecutor(max_workers=20)
    futures = []
    text_segments = []
    with open(src_text_file) as f_in:
        count=0
        buffer = []
        for line in tqdm.tqdm(f_in):
            no_punc_string = line.rstrip('\n').translate(str.maketrans('', '', string.punctuation))
            clean_string = " ".join(list(filter(None, no_punc_string.split(" "))))
            if count<100:
                buffer.append(clean_string)
                count+=1
            else:
                buffer.append(clean_string)
                text_list = buffer
                text_segments.append(text_list)
                count=0
                buffer=[]
                

    for text_list in text_segments:
        futures.append(
        executor.submit(normalize_parallel, normalizer,text_list)
        )

    results = sum([future.result() for future in futures],[])
    with open(dest_txt_file, 'w') as f_out:
        for line in results:
            f_out.write(line+'\n')


if __name__=='__main__':
    src_text_file = '/media/newhd/BookCorpus/books_large_p1.txt'
    dest_txt_file = '/media/newhd/BookCorpus/books_large_p1_normalized.txt'
    main(src_text_file, dest_txt_file)
    
