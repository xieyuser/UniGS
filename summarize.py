import json
import os
import sys
import numpy as np

def collect_values(data, previous_dict, data_root):
    for key, value in data["ours_30000"].items():
        if key in metrics_of_interest:
            all_nums = []
            for _, item in value.items():
                if isinstance(item, (int, float)):
                    all_nums.append(item)
            if all_nums:
                previous_dict[key] += all_nums
            if key == "PSNR":
                print(len(all_nums), "\t", data_root, "\t", np.mean(np.array(all_nums)))
    return previous_dict


def extract_numbers(data, num_list):
    if isinstance(data, dict):
        for value in data.values():
            extract_numbers(value, num_list)
    elif isinstance(data, list):
        for item in data:
            extract_numbers(item, num_list)
    elif isinstance(data, (int, float)):
        num_list.append(data)


def main(input_txt):
    with open(input_txt, "r") as f:
        json_paths = [line.strip() for line in f if line.strip()]

    aggregated_metrics = {metric: [] for metric in metrics_of_interest}

    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"Warning: File {json_path} does not exist. Skipping.")
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {json_path}. Skipping.")
            continue
        
        assert len(data["ours_30000"]) == len(metrics_of_interest), json_path


        aggregated_metrics = collect_values(data, aggregated_metrics, json_path)

    result = {}
    for metric, values in aggregated_metrics.items():
        if values:
            result[metric] = sum(values) / len(values)
        else:
            result[metric] = None
    print("total : ", len(values))
    for k, v in result.items():
        print(k,"\t",  v,"\t", f"{round(v, 3)}")


metrics_of_interest = [
    "SSIM",
    "PSNR",
    "LPIPS",
    "Depth_Abs_Rel",
    "Depth_Sq_Rel",
    "Depth_RMSE",
    "Depth_RMSE_Log",
    "Depth_A1",
    "Depth_A2",
    "Depth_A3",
    "Normal_Cosine",
    "Semantic_mIoU",
]

if __name__ == "__main__":
    main(sys.argv[1])

