import os
import python_minifier

source_dir = "./"
output_dir = "./"

file_name = input(
    "What is the relative path to the Python file you want to minify? "
).strip()

file_name = os.path.normpath(file_name)

file_found = False

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if not file.endswith(".py"):
            continue

        input_file_path = os.path.join(root, file)

        relative_file_path = os.path.normpath(
            os.path.relpath(input_file_path, source_dir)
        )

        if relative_file_path == file_name:
            file_found = True

            output_file_path = os.path.join(output_dir, "minified_" + relative_file_path)

            print(
                f"Minifying file: {relative_file_path}, "
                f"Saving to: {output_file_path}"
            )

            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

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

            print(f"Minified and saved: {output_file_path}")
            break

    if file_found:
        break

if not file_found:
    print(f"File not found: {file_name}")
