import multiprocessing
import runpy
import os

original_directory = os.getcwd()
def run_script(script):
    import subprocess
    subprocess.run(["python", script])

scripts = ["several_polymer_y.py","Several_y_0_05.py","Several_y_0_01.py","Several_y_0_005.py","Several_y_0_001.py"]


source_script = "several_polymer_y.py"
destination_scripts = ["Several_y_0_05.py","Several_y_0_01.py","Several_y_0_005.py","Several_y_0_001.py"]




for destination_script in destination_scripts:
    x = os.path.splitext(destination_script)[0]
    with open(source_script, "r") as src:
        lines = src.readlines()
    
    for i in range(len(lines)):
        if 'script_base_name = "several_polymer_y";' in lines[i]:
            lines[i] = f"script_base_name = \"{x}\";\n"
        if "src = script_dir + '/' + \"Model1\";" in lines[i]:
            lines[i] = f"src = script_dir + '/' + \"Model_{x}\";\n"
            
    with open(destination_script, "w") as dest:
        dest.writelines(lines)
    
    print(f"Copied content from {source_script} to {destination_script}.")
    print("="*80)

for destination_script in destination_scripts:
    # Print current working directory before running the script
    print(f"Before running {destination_script}, Current Working Directory: {os.getcwd()}")
    
    runpy.run_path(destination_script)  
    
    #go back to the original_directory
    os.chdir(original_directory)
     
    print("="*80)



