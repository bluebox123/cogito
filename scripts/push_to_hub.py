"""Push the final model + results + training recipe to the Hugging Face Hub.
Requires `huggingface-cli login` (a write token) beforehand."""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to final model dir")
    p.add_argument("--repo", required=True, help="e.g. your-username/cogito-3b")
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    p.add_argument("--private", action="store_true")
    a = p.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)

    print("[push] uploading model from %s" % a.model)
    api.upload_folder(folder_path=a.model, repo_id=a.repo, repo_type="model")

    if os.path.isdir(a.results):
        print("[push] uploading results/")
        api.upload_folder(folder_path=a.results, repo_id=a.repo, repo_type="model", path_in_repo="results")

    train_dir = os.path.join(ROOT, "train")
    if os.path.isdir(train_dir):
        print("[push] uploading recipe (train/*.py)")
        api.upload_folder(folder_path=train_dir, repo_id=a.repo, repo_type="model",
                          path_in_repo="recipe/train", allow_patterns=["*.py"])

    print("[push] DONE -> https://huggingface.co/%s" % a.repo)


if __name__ == "__main__":
    main()
