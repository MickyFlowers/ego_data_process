from supabase import create_client, Client
# Supabase 配置
url = "http://192.168.3.208:54321"
key = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"
supabase: Client = create_client(url, key)
 
def fetch_all_data(table_name, batch_size=1000):
 
    start = 0
    while True:
        # 获取 [start, start + batch_size - 1] 范围的数据
        response = (
            supabase.table(table_name)
            .select("path")
            .eq("dataset_name", "Ego4d")
            .not_.is_("id", "null")
            .range(start, start + batch_size - 1)
            .execute()
        )
        print(response.data[0]['path'].split('/')[-1].split('.')[0])
        if len(response.data) == 0:
            break
 
        start += len(response.data)
 
        print(f"已获取 {start} 条数据...")
    return start
 
if __name__ == "__main__":
    total_num = fetch_all_data(table_name="egocentric_dataset_clips")