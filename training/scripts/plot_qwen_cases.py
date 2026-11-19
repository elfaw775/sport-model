"""Plot reference boxes against Qwen's direct and adapted answers on the entire holdout."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from compare_detectors import WORK, truth
from score_qwen_vl import AXES, UNRESOLVED, load_qwen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direct_predictions", type=Path)
    parser.add_argument("adapted_predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    originals = WORK / "dataset/images/test"
    labels = truth("test")
    direct, adapted = (load_qwen(p)["test"] for p in (args.direct_predictions, args.adapted_predictions))
    assert set(direct) == set(adapted) == set(labels)
    styles = [("Reference", "#48d69c", "-"), ("Qwen direct", "#f5a94c", "--"),
              ("Qwen with QLoRA", "#92b7ff", ":")]
    figure, axes = plt.subplots(3, 4, figsize=(13, 10), facecolor="#f5f6f3")
    figure.subplots_adjust(left=.025, right=.975, top=.875, bottom=.025, hspace=.22, wspace=.12)
    figure.suptitle("Fetlock lesion localization on all 12 held-out films",
                    x=.025, y=.98, ha="left", fontsize=17, fontweight="bold")
    legend = [Line2D([], [], color=color, linestyle=style, linewidth=2.4, label=name)
              for name, color, style in styles]
    figure.legend(handles=legend, loc="upper left", bbox_to_anchor=(.018, .947), ncol=3, frameon=False)
    for index, (name, ax) in enumerate(zip(sorted(labels), axes.flat, strict=True), 1):
        with Image.open(originals / name) as image:
            width, height = image.size
            ax.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255,
                      extent=(0, width, height, 0), interpolation="bilinear")
        sets = [labels[name]]
        for result in (direct[name], adapted[name]):
            sets.append([tuple(box[k] for k in AXES) for box in result[1]])
        for boxes, (_, color, linestyle) in zip(sets, styles):
            for x1, y1, x2, y2 in boxes:
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                       edgecolor=color, linewidth=1.6, linestyle=linestyle))
        ax.set(xlim=(0, width), ylim=(height, 0))
        ax.axis("off")
        amounts = [str(len(labels[name]))]
        for status, boxes in (direct[name], adapted[name]):
            amounts.append(status.replace("_", " ") if status in UNRESOLVED else str(len(boxes)))
        counts = " / ".join(amounts)
        ax.set_title(f"{index:02d}    Boxes: {counts}", fontsize=10, loc="left")
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
