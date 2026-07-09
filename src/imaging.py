"""Small image helpers shared across the pipeline (resize + labeled panels)."""
from PIL import Image, ImageDraw, ImageFont


def resize_pil(x, size):
    return x.resize(size, resample=Image.LANCZOS)


def concat_images_with_labels(images, labels, path, colors=None):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                                  size=max(24, int(images[0].height * 0.05)))
    except Exception:
        font = ImageFont.load_default()

    labeled_images = []
    for i, (lbl, img) in enumerate(zip(labels[:len(images)], images)):
        img = img.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        fill = colors[i] if colors and i < len(colors) else "red"
        draw.text((10, 10), lbl, fill=fill, font=font)
        labeled_images.append(img)

    widths, heights = zip(*(i.size for i in labeled_images))
    total_width = sum(widths)
    max_height = max(heights)

    result = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in labeled_images:
        result.paste(img, (x_offset, 0))
        x_offset += img.width
    return result
