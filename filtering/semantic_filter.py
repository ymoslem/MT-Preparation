from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from tqdm import tqdm
import os
import sys

# Function to split source lines into chunks to avoid out-of-memory errors
def filter(source_file_path, target_file_path, model, pool, threshold, chunk_size, file_line_count):

    filtered_source = source_file_path + ".semantic"
    filtered_target = target_file_path + ".semantic"

    # Remove output files if exist
    if os.path.exists(filtered_source):
        os.remove(filtered_source)
    if os.path.exists(filtered_target):
        os.remove(filtered_target)


    # Open the files
    with open(source_file_path, "r") as source_file, open(target_file_path, "r") as target_file, \
    open(filtered_source, "a+") as filtered_source_file, open(filtered_target, "a+") as filtered_target_file:
        source = []
        target = []

        line_index = 0

        for src, tgt in zip(source_file, target_file):

            line_index += 1

            if line_index % chunk_size == 0 or line_index == file_line_count:
                print(line_index, "|", end=" ", flush=True)

                # Compute Sentence Embeddings
                source_embeddings = model.encode_multi_process(source, pool=pool, batch_size=2000)
                target_embeddings = model.encode_multi_process(target, pool=pool, batch_size=2000)

                # Find similar sentences (> threshold) and save to files
                index = 0

                for source_sentence_vecs, target_sentence_vecs in zip(source_embeddings, target_embeddings):
                    similarity = pytorch_cos_sim(source_sentence_vecs, target_sentence_vecs)

                    # Save to the source and target to files
                    if similarity > threshold:
                        filtered_source_file.write(source[index].strip() + "\n")
                        filtered_target_file.write(target[index].strip() + "\n")

                    index += 1

                source = []
                target = []

            else:
                source.append(src.strip())
                target.append(tgt.strip())


def line_count(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines


if __name__ == '__main__':

    # [Modify] source and target paths
    source_file_path = sys.argv[1]
    target_file_path = sys.argv[2]
    chunk_size = int(sys.argv[3])
    srclang = sys.argv[4]
    tgtlang = sys.argv[5]
    
    file_line_count = line_count(source_file_path)
    print("Line count:", file_line_count)


    # Download and load the model
    model_cache = "."
    
    if len(srclang) > 2 or len(tgtlang) > 2:
        raise SystemExit("Please use an ISO 639‑1 language code, e.g. 'en'!")
    elif srclang in muse_langs and tgtlang in muse_langs:
        model_name = "distiluse-base-multilingual-cased-v1"  # 15 languages
    elif srclang in para_langs and tgtlang in para_langs:
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # 50 languages
    else:
        raise SystemExit("Language pair is not supported!")
    
    model = SentenceTransformer(model_name, device="cuda", cache_folder=model_cache)
    print("Model loaded:", model_name)

    threshold = 0.45


    # Start a multiprocessing pool
    pool = model.start_multi_process_pool()

    # Filter
    filter(source_file_path, target_file_path, model, pool, threshold, chunk_size, file_line_count)

    # Close the the multiprocessing pool
    model.stop_multi_process_pool(pool)
