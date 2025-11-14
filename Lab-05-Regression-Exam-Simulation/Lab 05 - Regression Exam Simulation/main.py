import random
import numpy as np
import pandas as pd
import torch

SEED = 42
N = 100  

def main():
    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    indices = list(range(N))
    values = np.random.rand(N).tolist() 
    noise = torch.randn(N).numpy()   

    combined_values = [v + 0.1 * n for v, n in zip(values, noise)]

    df = pd.DataFrame({
        "index": indices,
        "value": combined_values
    })

    submission_path = "submission.csv"
    df.to_csv(submission_path, index=False)

if __name__ == "__main__":
    main()
