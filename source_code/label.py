import os
import csv
 
dataset_path = "/home/hai20521281/Downloads/TEST/dataset"
csv_file = "labels.csv"
 
def parse(name):
    tmp=name.split("-")[:-1]
    return "_".join(tmp)
 
truee=os.listdir("/home/hai20521281/Downloads/TEST/result_similar")
falsee=os.listdir("/home/hai20521281/Downloads/TEST/result_diff")

def classify(folder_name):
    if folder_name in truee:
        return True
    else:
        return False
def label(dataset_path,csv_file,truee,falsee):
    c1=0
    c0=0
    # Open the CSV file in write mode
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Folder", "Label"])  # Write the header row
   
        # Iterate through the folders in the dataset
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
       
            # Check if the folder name matches the label patterns
            if classify(folder_name):
                label = 1
                c1+=1
            else:
                label = 0
                c0+=1
       
            # Write the folder name and label to the CSV file
            writer.writerow([folder_name, label])
 
    print("done")
    print(c1)
    print(c0)


label(dataset_path,csv_file,truee,falsee)

