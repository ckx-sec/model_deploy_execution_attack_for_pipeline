import json
import argparse

def filter_hooks(input_path, output_path):
    """
    Filters a hook configuration file to REMOVE entries with 'b.eq' or 'b.ne' instructions.

    Args:
        input_path (str): The path to the input JSON file.
        output_path (str): The path where the filtered JSON file will be saved.
    """
    try:
        with open(input_path, 'r') as f:
            hooks_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析 '{input_path}' 中的 JSON 数据。")
        return

    if not isinstance(hooks_data, list):
        print("错误: JSON 文件的顶层结构应为一个列表。")
        return

    filtered_hooks = [
        hook for hook in hooks_data
        if hook.get("original_branch_instruction") not in ["b.eq", "b.ne"]
    ]

    try:
        with open(output_path, 'w') as f:
            json.dump(filtered_hooks, f, indent=4)
        print(f"成功! 已移除 'b.eq'/'b.ne' 条目，并将剩余的 {len(filtered_hooks)} 条记录写入到 '{output_path}'")
    except IOError as e:
        print(f"错误: 无法写入到文件 '{output_path}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="过滤 hook config JSON 文件，移除 b.eq 和 b.ne 指令。"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入的 hook config JSON 文件路径。"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出过滤后的 JSON 文件路径。"
    )

    args = parser.parse_args()
    filter_hooks(args.input, args.output) 