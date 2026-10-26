def padded_box(box, width, height, padding=0.1):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(width, int(x2 + dx)), min(height, int(y2 + dy))]
