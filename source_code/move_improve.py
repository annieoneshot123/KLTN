import subprocess

def copy_folders(source1, source2, destination):
    # Gọi lệnh rsync để sao chép từ source1 vào destination
    subprocess.call(["rsync", "-av", "--progress", source1, destination])

    # Gọi lệnh rsync để sao chép từ source2 vào destination
    subprocess.call(["rsync", "-av", "--progress", source2, destination])

# Gọi hàm copy_folders với các đường dẫn thư mục nguồn và đích

source1 = "/home/hai20521281/Downloads/TEST/result_similar_improve/"
source2 = "/home/hai20521281/Downloads/TEST/result_diff_improve/"
destination = "/home/hai20521281/Downloads/TEST/dataset_improve/"
copy_folders(source1, source2, destination)

