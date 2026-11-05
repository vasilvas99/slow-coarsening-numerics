import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

parser = argparse.ArgumentParser()
parser.add_argument(
    "INPUT_TRAJECTORY",
    type=str,
    help="The path to the input trajectory",
)
parser.add_argument("OUTPUT_VIDEO", help="The output video path")
(parser.add_argument("--fps", type=int, default=24, help="Video fps, default 24."),)
parser.add_argument("--dpi", type=int, default=180, help="Image DPI, default 180.")
args = parser.parse_args()


def render_video(trajectory_file, video_file, fps, dpi):
    data = np.load(trajectory_file)

    times = data["t"]
    states = data["u"]
    x = data["x"]
    n_steps = int(data["N"])
    l0 = float(data["l0"])

    frame_times = np.unique(
        np.concatenate(
            [
                np.linspace(0, 0.3, 180),
                np.geomspace(0.3, 3, 60),
                np.geomspace(3, times[-1], 90),
            ]
        )
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.arange(n_steps))

    lines = []
    for n in range(n_steps):
        (line,) = ax.plot(x, states[0, n], color=colors[n], linewidth=1.8)
        lines.append(line)

        ax.axhline(
            (n + 1) * l0,
            color="gray",
            linestyle=":",
            linewidth=0.6,
            alpha=0.4,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(states.min() - 0.5, states.max() + 0.5)
    ax.set_xlabel("$x$", fontsize=22)
    ax.set_ylabel("$u_n$", fontsize=22)
    ax.tick_params(labelsize=18)

    time_label = ax.text(
        0.98,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=18,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.85,
        },
    )

    fig.tight_layout()

    writer = FFMpegWriter(
        fps=fps,
        bitrate=2500,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )

    with writer.saving(fig, video_file, dpi=dpi):
        for frame, t in enumerate(frame_times):
            # Find the two saved states surrounding the frame time
            right = np.searchsorted(times, t, side="right")
            right = np.clip(right, 1, len(times) - 1)
            left = right - 1

            # Linearly interpolate between the saved states
            fraction = (t - times[left]) / (times[right] - times[left])
            u = (1 - fraction) * states[left] + fraction * states[right]

            # Update step positions
            for n in range(n_steps):
                lines[n].set_ydata(u[n])

            if t < 1:
                time_text = f"{t:.3f}"
            elif t < 100:
                time_text = f"{t:.2f}"
            else:
                time_text = f"{t:.2e}"

            time_label.set_text(f"$t = {time_text}$")
            writer.grab_frame()

            if frame % 50 == 0:
                print(f"Frame {frame + 1}/{len(frame_times)}")

    plt.close(fig)
    print(f"Saved {video_file}")


render_video(args.INPUT_TRAJECTORY, args.OUTPUT_VIDEO, args.fps, args.dpi)
