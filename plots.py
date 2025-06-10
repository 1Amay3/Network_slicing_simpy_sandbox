import pandas as pd
import sys,os
import matplotlib.pyplot as plt

def plot_from_csv(path):
    df=pd.read_csv(path)

    slices = df["Slice"].unique()
    fix,ax=plt.subplots(len(slices),1,figsize=(10,4*len(slices)),sharex=True)


    for i, sl in enumerate(sorted(slices)):
        sub = df[df["Slice"] == sl]
        ax[i].plot(sub["Time"], sub["Asked"],    label="Asked")
        ax[i].plot(sub["Time"], sub["Admitted"], label="Admitted")
        ax[i].plot(sub["Time"], sub["Dropped"],  label="Dropped")
        ax[i].set_ylabel(f"{sl}  (Mbps)")
        ax[i].legend(loc="upper right")

    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "logs/simulation_log.csv"
    if not os.path.exists(csv_path):
        print(f"No such file: {csv_path}")
        sys.exit(1)

    plot_from_csv(csv_path)
