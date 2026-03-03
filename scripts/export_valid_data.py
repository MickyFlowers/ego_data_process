from supabase import create_client, Client
# Supabase 配置
url = "http://192.168.3.208:54321"
key = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"
supabase: Client = create_client(url, key)
 
def fetch_all_data(table_name, dataset_name, batch_size=1000):
 
    start = 0
    all_clip_ids = []
    while True:
        # 获取 [start, start + batch_size - 1] 范围的数据
        response = (
            supabase.table(table_name)
            .select("path")
            .eq("dataset_name", dataset_name)
            .not_.is_("pose3d_hand_path", "null")
            # .is_("multi_hand_flag", "False")
            .range(start, start + batch_size - 1)
            .execute()
        )
        if len(response.data) == 0:
            break
        clip_ids = [response.data[i]['path'].replace('oss://', "/home/").split('/')[-1].split('.')[0] for i in range(len(response.data))]
        all_clip_ids += clip_ids
        start += len(response.data)
 
        print(f"已获取 {start} 条数据...")

    return all_clip_ids
 
if __name__ == "__main__":
    all_clip_ids = fetch_all_data(table_name="egocentric_dataset_clips", dataset_name="ml-egodex")
    import json
    with open("/home/ss-oss1/data/dataset/egocentric/ml-egodex/valid_data.json", "w") as f:
        json.dump(all_clip_ids, f, indent=10)
