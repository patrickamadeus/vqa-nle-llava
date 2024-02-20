def unpack_json(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{json_file_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def inference_llava(model, processor, prompt, img, max_new_tokens=500, do_sample=False, skip_special_tokens=True) -> str:
    complete_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    
    inputs = processor(
        complete_prompt, 
        img, 
        return_tensors = 'pt'
    ).to(0, torch.float16)
    
    raw_output = model.generate(
        **inputs, 
        max_new_tokens = max_new_tokens, 
        do_sample = do_sample
    )
    
    output = processor.decode(raw_output[0], skip_special_tokens = skip_special_tokens)
    output_trunc = output[output.index("ASSISTANT:") + 11:]
    
    return output_trunc

def exec_time(to, tt) -> str:
    time_difference = tt - to

    hours, remainder = divmod(time_difference.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    result_format = f"{hours}h{minutes}m{seconds}s"
    
    return result_format
