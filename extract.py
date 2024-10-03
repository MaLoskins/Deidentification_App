# extract.py

import os
import argparse

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Search for files with specific formats and write their contents to a text file.'
    )

    parser.add_argument(
        '-d', '--directory',
        type=str,
        default=r"C:\Users\matth\OneDrive\Desktop\1. DATA SCIENCE MASTER\Capstone_CITS5553\Application",
        help='The directory to search. Default is "C:\\Users\\matth\\OneDrive\\Desktop\\1. DATA SCIENCE MASTER\\Research_CITS5014\\MY_OWN_GAT"'
    )

    parser.add_argument(
        '-f', '--formats',
        type=str,
        nargs='+',
        default=[".py"],
        help='One or more file formats to search for (e.g., .py .txt .ipynb). Default is ".py"'
    )

    parser.add_argument(
        '-i', '--ignore',
        type=str,
        nargs='*',
        default=[],
        help='One or more directories to ignore. Default is to always ignore ".venv"'
    )

    return parser.parse_args()

def main():
    """
    Main function to execute the script logic.
    """
    args = parse_arguments()
    search_dir = args.directory
    file_formats = set(args.formats)
    
    # Always ignore '.venv' plus any additional directories specified
    ignore_dirs = set(args.ignore) | {'.venv'}

    # Path to the output file
    output_file = os.path.join(search_dir, "FileName_Contents.txt")

    # Clear the output file if it exists
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # This will clear the file
    except Exception as e:
        print(f"Error initializing the output file: {e}")
        return

    # Walk through the directory
    for root, dirs, files in os.walk(search_dir, topdown=True):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            if any(file.endswith(ext) for ext in file_formats) and file != "extract.py":
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                # Write the file name and its contents to the output file
                try:
                    with open(output_file, 'a', encoding='utf-8') as out_f:
                        out_f.write(f'"{file}"\n\n```\n{content}\n```\n\n')
                except Exception as e:
                    print(f"Error writing to {output_file}: {e}")
                    return

    print(f"Contents have been successfully written to {output_file}")

if __name__ == "__main__":
    main()
