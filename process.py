from huggingface_hub import snapshot_download
import subprocess

repo_name = "gemma-2-9b-it_gsm8k_ent0.05_beam5_dosampleFalse_temp0.8_estep_"
snapshot_download(repo_id=f"YYT-t/{repo_name}", 
                  local_dir=f"Q_models/{repo_name}")
subprocess.run([
    "huggingface-cli", "upload", 
    f"YYT-t/{repo_name}_final", 
    f"Q_models/{repo_name}/final_ckpt", 
    "--token", "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"
])
# store it