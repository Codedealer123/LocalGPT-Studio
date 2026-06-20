import os
import shutil
import python_minifier

source_dir = "./backend"
output_dir = "./prodBackend"

for root, dirs, files in os.walk(source_dir):
    for file in files:
        input_file_path = os.path.join(root, file)

        relative_path = os.path.relpath(root, source_dir)
        target_folder = os.path.join(output_dir, relative_path)
        output_file_path = os.path.join(target_folder, file)

        if file.endswith(".py"):
            print(
                "Minifying file:",
                file,
                ", Saving to:",
                output_file_path
            )

            os.makedirs(target_folder, exist_ok=True)

            with open(input_file_path, "r", encoding="utf-8") as f:
                original_code = f.read()

            minified_code = python_minifier.minify(
                original_code,
                combine_imports=True,
                remove_pass=True,
                remove_literal_statements=True,
                hoist_literals=True,
                rename_locals=True,
                preserve_locals=[],
                rename_globals=False,
                preserve_globals=["handler"],
                convert_posargs_to_args=True,
                preserve_shebang=True,
                remove_asserts=True,
                remove_debug=True,
                remove_explicit_return_none=True,
                remove_builtin_exception_brackets=True,
                constant_folding=True,
                remove_annotations=True,
            )

            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(minified_code)

            print(f"Minified and saved: {file}")

        else:
            print(f"Copying asset file: {file}")

            os.makedirs(target_folder, exist_ok=True)
            shutil.copy2(input_file_path, output_file_path)

            print(f"Copied asset file: {file}")
