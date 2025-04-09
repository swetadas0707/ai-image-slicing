from grid_based_slicing import grid_slicing
from projection_based_slicing import projection_slicing

def slice(img_path, page_type, output_folder):
    if page_type == "grid":
        grid_slicing(img_path, output_folder)
    elif page_type == "projection":
        projection_slicing(img_path, output_folder)
    else:
        print("Invalid page type. Choose 'grid' or 'projection'.")

if __name__ == "__main__":
    
    slice("/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/pics/bts.jpeg", page_type="grid", output_folder="grid_bts")
