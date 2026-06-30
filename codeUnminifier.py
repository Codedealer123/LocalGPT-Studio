import ast
import os
import shutil

source_dir = "./prodBackend"
output_dir = "./backend"

# Clean previous output
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        input_file_path = os.path.join(root, file)

        relative_path = os.path.relpath(root, source_dir)
        target_folder = os.path.join(output_dir, relative_path)
        output_file_path = os.path.join(target_folder, file)

        if file.endswith(".py"):
            print(
                "Formatting file:",
                file,
                ", Saving to:",
                output_file_path,
            )

            os.makedirs(target_folder, exist_ok=True)

            with open(input_file_path, "r", encoding="utf-8") as f:
                code = f.read()

            try:
                tree = ast.parse(code)
                formatted_code = ast.unparse(tree)

                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(formatted_code + "\n")

                print(f"Formatted and saved: {file}")

            except SyntaxError as e:
                print(f"Failed to parse {file}: {e}")
                shutil.copy2(input_file_path, output_file_path)

        else:
            print(f"Copying asset file: {file}")

            os.makedirs(target_folder, exist_ok=True)
            shutil.copy2(input_file_path, output_file_path)

            print(f"Copied asset file: {file}")