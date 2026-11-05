# Draws docs/figures/designs.svg: where a conversation's second turn goes under each design.
# Standard library only.   python3 docs/make_diagram.py
# Times are calibrated costs for an 8k-token prefix on one A100 (Qwen2.5-7B, calibrate.py).
import os

INK, INK2, EDGE, RED, GREEN = "#1f2328", "#57606a", "#8c959f", "#9c4a4a", "#3f8a55"
REPLICA, KV, TIER, REQ = "#eef2f7", "#f2dcdc", "#d6eadb", "#d8e1ef"
PW, H = 320, 330          # panel width, canvas height
A, B, ROW = 30, 175, 150  # replica A / B x offsets within a panel, replica row y


def text(x, y, s, size=12, color=INK, weight="normal"):
    return (f"<text x='{x}' y='{y}' font-family='Helvetica, Arial, sans-serif' font-size='{size}' "
            f"text-anchor='middle' fill='{color}' font-weight='{weight}'>{s}</text>")


def box(x, y, w, h, fill):
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='5' fill='{fill}' stroke='{EDGE}' stroke-width='1.2'/>"


def arrow(x1, y1, x2, y2, color=INK2, dash=False):
    d = " stroke-dasharray='5 4'" if dash else ""
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.6'{d} "
            f"marker-end='url(#head-{color[1:]})'/>")


def panel(x0, title, subtitle, note, note_color):
    """Title, router, and two replicas; replica A already holds turn 1's KV."""
    a, b = x0 + A, x0 + B
    return [text(x0 + PW / 2, 28, title, 14, weight="bold"), text(x0 + PW / 2, 46, subtitle, 11, INK2),
            box(x0 + 110, 62, 100, 30, "#ffffff"), text(x0 + 160, 82, "router"),
            box(a, ROW, 115, 62, REPLICA), text(a + 57, ROW + 18, "replica A", 12, weight="bold"),
            box(a + 12, ROW + 28, 91, 22, KV), text(a + 57, ROW + 43, "turn 1 KV", 11),
            box(b, ROW, 115, 62, REPLICA), text(b + 57, ROW + 18, "replica B", 12, weight="bold"),
            text(x0 + PW / 2, H - 10, note, 11, note_color, "bold")]


def affinity(x0):
    a, b = x0 + A, x0 + B
    out = panel(x0, "Prefix affinity", "turn 2 goes where turn 1's KV is",
                "same system prompt everywhere → one hot replica", RED)
    out += [arrow(x0 + 140, 92, a + 57, ROW - 26), text(b + 57, ROW + 43, "idle", 11, INK2)]
    out += [box(a + 8 + 11 * i, ROW - 22, 8, 14, REQ) for i in range(9)]  # its queue
    out.append(text(a + 57, ROW + 80, "every request queues here", 10, INK2))
    return out


def least(x0):
    b = x0 + B
    out = panel(x0, "Least-loaded", "turn 2 goes to the less busy replica",
                "recompute 8k tokens: 0.57 s", RED)
    return out + [arrow(x0 + 180, 92, b + 57, ROW - 2), text(b + 57, ROW + 43, "turn 2: recompute", 11, RED)]


def shared(x0):
    a, b = x0 + A, x0 + B
    out = panel(x0, "Least-loaded + shared tier", "any replica can read any prefix",
                "load 8k tokens from the tier: 0.14 s", GREEN)
    return out + [arrow(x0 + 180, 92, b + 57, ROW - 2), text(b + 57, ROW + 43, "turn 2: load", 11, GREEN),
                  box(x0 + 30, 250, 260, 32, TIER), text(x0 + 160, 271, "shared KV tier (host memory)", 11),
                  arrow(a + 57, ROW + 62, a + 57, 248, INK2, dash=True), text(a + 80, 236, "write", 10, INK2),
                  arrow(b + 57, 250, b + 57, ROW + 64, GREEN), text(b + 80, 236, "read", 10, GREEN)]


def main():
    heads = "".join(f"<marker id='head-{c[1:]}' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'>"
                    f"<path d='M0,0 L8,4 L0,8 z' fill='{c}'/></marker>" for c in [INK2, GREEN])
    body = [f"<defs>{heads}</defs>", f"<rect width='{3 * PW}' height='{H}' fill='white'/>"]
    body += affinity(0) + least(PW) + shared(2 * PW)
    body += [f"<line x1='{x}' y1='16' x2='{x}' y2='{H - 30}' stroke='#d0d7de'/>" for x in [PW, 2 * PW]]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "designs.svg")
    with open(out, "w") as f:
        f.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='{3 * PW}' height='{H}' "
                f"viewBox='0 0 {3 * PW} {H}'>" + "".join(body) + "</svg>")
    print("wrote", out)


if __name__ == "__main__":
    main()
