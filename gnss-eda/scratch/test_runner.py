import nbformat
from nbclient import NotebookClient
import sys


def run_notebook(nb_path):
    print(f"Reading {nb_path}...")
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    print(f"Executing {nb_path}...")
    client.execute()
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    print(f"Finished {nb_path} successfully.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_notebook(sys.argv[1])
