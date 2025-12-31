from google.cloud import bigquery
import os

def list_all_bigquery_objects():
    print("🔍 Listing all BigQuery objects with explicit credentials...")
    sa_path = "Service Account BigQuery/sa-gatra-bigquery.json"
    if os.path.exists(sa_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        print(f"✅ Using SA: {sa_path}")
    
    client = bigquery.Client(project="chronicle-dev-2be9")
    
    datasets = ["gatra_database", "soc_data"]
    
    for ds_id in datasets:
        print(f"\n📂 Dataset: {ds_id}")
        dataset_id = f"chronicle-dev-2be9.{ds_id}"
        try:
            tables = list(client.list_tables(dataset_id))
            if tables:
                for table in tables:
                    print(f"  - [{table.table_type}] {table.table_id}")
            else:
                print("  (No tables found)")
        except Exception as e:
            print(f"  ❌ Error listing tables: {e}")

if __name__ == "__main__":
    list_all_bigquery_objects()
