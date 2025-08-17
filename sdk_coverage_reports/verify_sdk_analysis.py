import csv
from collections import defaultdict

def analyze_sdk_coverage(filepath):
    """
    Analyzes SDK coverage from the sdk_analysis.csv file.

    Args:
        filepath (str): The path to the sdk_analysis.csv file.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    sdk_columns = [
        "SenseTime SDK",
        "Face++ SDK",
        "Arcsoft SDK",
        "Biometric SensorTech SDK"
    ]

    # Initialize data structures
    model_counts = defaultdict(int)
    # Using a set to automatically handle unique brand names
    brands_per_sdk = defaultdict(set)
    # Nested defaultdict to store model counts per brand for each sdk
    brand_model_counts = defaultdict(lambda: defaultdict(int))

    try:
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                brand = row.get("Brand")
                if not brand:
                    continue

                for sdk in sdk_columns:
                    # If the SDK column has a value, it means the SDK is present
                    if row.get(sdk, "").strip():
                        model_counts[sdk] += 1
                        brands_per_sdk[sdk].add(brand)
                        brand_model_counts[sdk][brand] += 1
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    # --- Process the collected data ---
    results = {}
    for sdk in sdk_columns:
        total_models = model_counts[sdk]
        total_brands = len(brands_per_sdk[sdk])

        # Sort brands by model count in descending order
        sorted_brands = sorted(
            brand_model_counts[sdk].items(),
            key=lambda item: item[1],
            reverse=True
        )

        # Get top 6 (or all if fewer than 6)
        top_brands_data = []
        for brand, count in sorted_brands[:6]:
            percentage = (count / total_models) * 100 if total_models > 0 else 0
            top_brands_data.append({
                "brand": brand,
                "count": count,
                "share": round(percentage, 1)
            })

        results[sdk] = {
            "total_brands": total_brands,
            "total_models": total_models,
            "top_brands": top_brands_data
        }
    return results

def print_markdown_table(analysis_results):
    """Prints the analysis results as a markdown table."""
    # Header
    print("| SDK Provider | 覆盖品牌数 (Brands) | 覆盖设备型号数 (Models) | 主要贡献品牌 (Top 6 Brands) | 品牌贡献详情 (Models Count, Share) |")
    print("| :--- | :--- | :--- | :--- | :--- |")

    # Rows
    for sdk, data in analysis_results.items():
        provider_name = sdk.replace(" SDK", "")
        total_brands = data['total_brands']
        total_models = data['total_models']

        top_brands_list = [item['brand'] for item in data['top_brands']]
        # Abbreviate long names
        top_brands_str = ", ".join(b if b != "deutschetelekom" else "D.Telekom" for b in top_brands_list)


        breakdown_list = [
            f"{item['brand'] if item['brand'] != 'deutschetelekom' else 'D.Telekom'} ({item['count']}, **{item['share']}%**)"
            for item in data['top_brands']
        ]
        breakdown_str = ", ".join(breakdown_list)

        print(f"| **{provider_name}** | {total_brands} | {total_models} | {top_brands_str} | {breakdown_str} |")


if __name__ == "__main__":
    csv_file_path = 'sdk_analysis.csv'
    final_results = analyze_sdk_coverage(csv_file_path)

    if final_results:
        print_markdown_table(final_results) 