"""Image preparation, drawing, and copy helpers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageStat

from .vcot_branch import annotate_image_with_ticks

from .config import QWEN_INPUT_MAX_SIDE, QWEN_MULTI_IMAGE_SIDE_SEQUENCE, Point


FLUORESCENT_BOX_COLORS: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
	("fluorescent_red", (255, 0, 76)),
	("fluorescent_yellow", (255, 255, 0)),
	("fluorescent_blue", (0, 128, 255)),
	("fluorescent_green", (57, 255, 20)),
)

def copy_image(src: Path, dst: Path) -> Path:
	"""Copy an image with metadata unless source and destination are identical."""
	dst.parent.mkdir(parents=True, exist_ok=True)
	if src.resolve() != dst.resolve():
		shutil.copy2(src, dst)
	return dst


def normalize_image_max_side(image: Image.Image, max_side: Optional[int], upscale: bool = False) -> Image.Image:
	"""Resize by long side while preserving aspect ratio.

	By default small images are not enlarged.  Returning the original object
	when no resize is needed avoids unnecessary re-encoding by callers that only
	inspect the result.
	"""
	if max_side is None or max_side <= 0:
		return image
	long_side = max(image.size)
	if long_side <= 0 or long_side == max_side:
		return image
	if long_side < max_side and not upscale:
		return image
	scale = max_side / float(long_side)
	new_size = (
		max(1, int(round(image.width * scale))),
		max(1, int(round(image.height * scale))),
	)
	if new_size == image.size:
		return image
	return image.resize(new_size, Image.Resampling.LANCZOS)


def copy_image_normalized(src: Path, dst: Path, max_side: Optional[int], upscale: bool = False) -> Path:
	"""Decode to RGB, normalize the long side, and save a standalone artifact."""
	dst.parent.mkdir(parents=True, exist_ok=True)
	with Image.open(src) as loaded:
		image = loaded.convert("RGB")
		normalized = normalize_image_max_side(image, max_side, upscale=upscale)
		normalized.save(dst)
	return dst


def generation_resolution(args: argparse.Namespace) -> int:
	"""Read and validate the square output resolution from CLI arguments."""
	resolution = int(getattr(args, "resolution", 1024))
	if resolution <= 0:
		raise ValueError(f"--resolution must be positive, got {resolution}")
	return resolution


def resize_image_to_generation_resolution(image: Image.Image, args: argparse.Namespace) -> Image.Image:
	"""Return an image resized to the generator's square output contract."""
	resolution = generation_resolution(args)
	target_size = (resolution, resolution)
	if image.size == target_size:
		return image
	return image.resize(target_size, Image.Resampling.LANCZOS)


def copy_image_generation_sized(src: Path, dst: Path, args: argparse.Namespace) -> Path:
	"""Materialize an RGB copy at the configured generation resolution."""
	dst.parent.mkdir(parents=True, exist_ok=True)
	with Image.open(src) as loaded:
		image = loaded.convert("RGB")
		resized = resize_image_to_generation_resolution(image, args)
		resized.save(dst)
	return dst


def load_font(size: int) -> Any:
	"""Load a predictable bold font, falling back to Pillow's bundled font."""
	for candidate in (
		"/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
		"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
	):
		try:
			return ImageFont.truetype(candidate, size)
		except OSError:
			continue
	return ImageFont.load_default()


def prepare_qwen_image(image: Image.Image, max_side: int = QWEN_INPUT_MAX_SIDE) -> Image.Image:
	"""Create an aspect-preserving Qwen input without upscaling small images."""
	if max_side <= 0:
		return image.copy()
	width, height = image.size
	long_side = max(width, height)
	if long_side <= max_side:
		return image.copy()
	scale = max_side / float(long_side)
	new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
	return image.resize(new_size, Image.Resampling.LANCZOS)


def pad_image_to_multiple(image: Image.Image, multiple: int = 32, fill: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
	"""Pad the right and bottom edges without changing image-space coordinates."""
	if multiple <= 1:
		return image.copy()
	width, height = image.size
	padded_width = ((width + multiple - 1) // multiple) * multiple
	padded_height = ((height + multiple - 1) // multiple) * multiple
	if (padded_width, padded_height) == image.size:
		return image.copy()
	padded = Image.new("RGB", (padded_width, padded_height), fill)
	padded.paste(image.convert("RGB"), (0, 0))
	return padded


def annotate_image_with_ticks_32(image: Image.Image, step: int = 128) -> Image.Image:
	"""Add the coordinate ruler used by L1 planning, then satisfy 32px alignment."""
	return pad_image_to_multiple(annotate_image_with_ticks(image, step=step), multiple=32)


def _srgb_channel_to_linear(value: float) -> float:
	"""Convert one 0-255 sRGB channel to linear-light intensity."""
	value = max(0.0, min(1.0, float(value) / 255.0))
	return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb: Sequence[float]) -> Tuple[float, float, float]:
	"""Convert an sRGB color to CIE Lab (D65) for perceptual distance."""

	r, g, b = (_srgb_channel_to_linear(float(value)) for value in rgb[:3])
	x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
	y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
	z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
	delta = 6.0 / 29.0

	def pivot(value: float) -> float:
		"""Apply the piecewise CIE Lab transfer function."""
		return value ** (1.0 / 3.0) if value > delta**3 else value / (3.0 * delta**2) + 4.0 / 29.0

	fx, fy, fz = pivot(x), pivot(y), pivot(z)
	return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def perceptual_color_distance(rgb_a: Sequence[float], rgb_b: Sequence[float]) -> float:
	"""Return Euclidean CIE-Lab distance between two sRGB triples."""
	lab_a = rgb_to_lab(rgb_a)
	lab_b = rgb_to_lab(rgb_b)
	return math.sqrt(sum((left - right) ** 2 for left, right in zip(lab_a, lab_b)))


def _edit_box_xyxy(edit: Dict[str, Any], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
	"""Convert a valid four-point edit quad to a clipped half-open box."""
	points = []
	for point in edit.get("quad_points") or []:
		if not isinstance(point, (list, tuple)) or len(point) < 2:
			continue
		try:
			points.append((int(round(float(point[0]))), int(round(float(point[1])))))
		except (TypeError, ValueError):
			continue
	if len(points) != 4:
		return None
	x0 = max(0, min(width - 1, min(point[0] for point in points)))
	y0 = max(0, min(height - 1, min(point[1] for point in points)))
	x1 = max(x0 + 1, min(width, max(point[0] for point in points) + 1))
	y1 = max(y0 + 1, min(height, max(point[1] for point in points) + 1))
	return x0, y0, x1, y1


def choose_fluorescent_box_color(image: Image.Image, box_xyxy: Sequence[int]) -> Dict[str, Any]:
	"""Pick the fluorescent color farthest from the mean color under a box."""

	x0, y0, x1, y1 = (int(value) for value in box_xyxy[:4])
	region = image.convert("RGB").crop((x0, y0, x1, y1))
	mean_rgb = tuple(float(value) for value in ImageStat.Stat(region).mean[:3])
	scored = [
		(name, rgb, perceptual_color_distance(mean_rgb, rgb))
		for name, rgb in FLUORESCENT_BOX_COLORS
	]
	name, rgb, distance = max(scored, key=lambda item: item[2])
	return {
		"name": name,
		"rgb": list(rgb),
		"covered_mean_rgb": [round(value, 3) for value in mean_rgb],
		"lab_distance": round(distance, 3),
		"candidate_distances": {item_name: round(item_distance, 3) for item_name, _item_rgb, item_distance in scored},
	}


def draw_dynamic_contrast_boxes(
	image: Image.Image,
	edits: Sequence[Dict[str, Any]],
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
	"""Draw each proposed edit bbox on the full image with a contrasting color."""

	base = image.convert("RGB")
	annotated = base.copy()
	draw = ImageDraw.Draw(annotated)
	line_width = max(5, int(round(min(base.size) / 160.0)))
	font = load_font(max(18, int(round(min(base.size) / 45.0))))
	records: List[Dict[str, Any]] = []
	for index, edit in enumerate(edits, start=1):
		box = _edit_box_xyxy(edit, base.width, base.height)
		if box is None:
			continue
		color_record = choose_fluorescent_box_color(base, box)
		color = tuple(int(value) for value in color_record["rgb"])
		x0, y0, x1, y1 = box
		draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=line_width)
		label = f"{index}:{edit.get('id') or f'edit_{index}'}"
		text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
		text_width = text_box[2] - text_box[0]
		text_height = text_box[3] - text_box[1]
		label_x = min(max(0, x0), max(0, base.width - text_width - 10))
		label_y = y0 - text_height - 10 if y0 >= text_height + 10 else min(base.height - text_height - 10, y0 + line_width + 2)
		label_y = max(0, label_y)
		draw.rectangle(
			(label_x, label_y, label_x + text_width + 10, label_y + text_height + 8),
			fill=(0, 0, 0),
			outline=color,
			width=max(2, line_width // 2),
		)
		draw.text((label_x + 5, label_y + 3), label, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
		records.append(
			{
				"edit_index": index,
				"id": edit.get("id") or f"edit_{index}",
				"box_xyxy": list(box),
				"line_width": line_width,
				**color_record,
			}
		)
	return annotated, records


def build_box_review_edit_lines(edits: Sequence[Dict[str, Any]]) -> str:
	"""Serialize proposed regions into stable, human-readable review lines."""
	lines: List[str] = []
	for idx, edit in enumerate(edits, start=1):
		lines.append(
			(
				f"{idx}. id={edit.get('id') or f'edit_{idx}'}; "
				f"target={edit.get('target_object') or 'Unknown'}; "
				f"issue={edit.get('issue') or 'Unknown'}; "
				f"desired={edit.get('edit_prompt') or edit.get('whole_edit') or 'Unknown'}; "
				f"quad_points={json.dumps(edit.get('quad_points') or [], ensure_ascii=False)}"
			)
		)
	return "\n".join(lines)


def qwen_multi_image_ladder(max_side: int) -> List[int]:
	"""Build the descending, de-duplicated image-size ladder for OOM retries."""
	start = max(1, int(max_side))
	ladder = [side for side in QWEN_MULTI_IMAGE_SIDE_SEQUENCE if side <= start]
	if not ladder:
		ladder = [start]
	elif ladder[0] != start:
		ladder.insert(0, start)
	return sorted(set(ladder), reverse=True)


def next_ladder_side(current_side: int, ladder: Sequence[int]) -> int:
	"""Return the next smaller configured side, or the current side at the floor."""
	for side in sorted({int(item) for item in ladder if int(item) > 0}, reverse=True):
		if side < current_side:
			return side
	return current_side


def load_qwen_candidate_images(candidates: Sequence[Tuple[str, Path]], max_side: int) -> List[Image.Image]:
	"""Load comparison candidates in label order and resize them for Qwen."""
	images: List[Image.Image] = []
	for _label, path in candidates:
		with Image.open(path) as loaded:
			images.append(prepare_qwen_image(loaded.convert("RGB"), max_side=max_side))
	return images


def build_labeled_multi_image_messages(
	system_prompt: str,
	user_text: str,
	candidates: Sequence[Tuple[str, Path]],
	max_side: int,
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
	"""Build labeled multimodal messages and return their owned PIL images.

	The separate image list gives callers explicit ownership so they can close
	decoded files after inference.  Labels are inserted directly before their
	corresponding image to keep blind A/B role mapping unambiguous.
	"""
	images = load_qwen_candidate_images(candidates, max_side=max_side)
	content: List[Dict[str, Any]] = []
	for (label, _path), image in zip(candidates, images):
		content.append({"type": "text", "text": f"{label}:"})
		content.append({"type": "image", "image": image})
	content.append({"type": "text", "text": user_text})
	return [
		{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
		{"role": "user", "content": content},
	], images


def scale_quad_points(quad_points: Sequence[Sequence[int]], scale_x: float, scale_y: float) -> List[List[int]]:
	"""Scale a four-point region between image coordinate systems."""
	return [
		[
			int(round(float(point[0]) * scale_x)),
			int(round(float(point[1]) * scale_y)),
		]
		for point in quad_points
	]


def set_prepared_qwen_image(item: Dict[str, Any], max_side: int) -> None:
	"""Rebuild one queued Qwen job at a smaller side for an OOM retry.

	Multi-image jobs use their stored message builder so every candidate is
	resized consistently.  Single-image jobs replace all message references with
	the newly prepared PIL object.
	"""
	builder = item.get("qwen_messages_builder")
	if callable(builder):
		messages, images = builder(max_side)
		item["messages"] = messages
		item["images"] = images
		item["qwen_image"] = images[0] if len(images) == 1 else None
		item["current_qwen_max_side"] = max((max(image.size) for image in images), default=max_side)
		return
	source_image = item.get("source_qwen_image") or item.get("qwen_image")
	if source_image is None:
		return
	qwen_image = prepare_qwen_image(source_image, max_side=max_side)
	item["qwen_image"] = qwen_image
	item["images"] = [qwen_image]
	for message in item.get("messages", []):
		for content in message.get("content", []):
			if isinstance(content, dict) and content.get("type") == "image":
				content["image"] = qwen_image
	item["current_qwen_max_side"] = max(qwen_image.size)


def draw_labeled_collage(
	images: Sequence[Tuple[str, Path]],
	grid_cols: int,
	output_path: Path,
	precompress_max_side: Optional[int] = None,
) -> Path:
	"""Render labeled candidates on a regular grid for offline inspection."""
	loaded = []
	for label, path in images:
		image = Image.open(path).convert("RGB")
		if precompress_max_side is not None:
			image = prepare_qwen_image(image, max_side=precompress_max_side)
		loaded.append((label, image))
	max_width = max(image.width for _, image in loaded)
	max_height = max(image.height for _, image in loaded)
	title_height = 44
	pad = 16
	grid_rows = (len(loaded) + grid_cols - 1) // grid_cols
	canvas_width = grid_cols * max_width + (grid_cols + 1) * pad
	canvas_height = grid_rows * (max_height + title_height) + (grid_rows + 1) * pad
	canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 245, 245))
	draw = ImageDraw.Draw(canvas)
	font = load_font(22)
	for idx, (label, image) in enumerate(loaded):
		row = idx // grid_cols
		col = idx % grid_cols
		x = pad + col * (max_width + pad)
		y = pad + row * (max_height + title_height + pad)
		draw.rectangle((x, y, x + max_width, y + title_height - 8), fill=(255, 255, 255), outline=(180, 180, 180))
		draw.text((x + 10, y + 8), label, fill=(0, 0, 0), font=font)
		px = x + (max_width - image.width) // 2
		py = y + title_height
		canvas.paste(image, (px, py))
		draw.rectangle((x, y + title_height, x + max_width, y + title_height + max_height), outline=(180, 180, 180), width=2)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	canvas.save(output_path)
	return output_path
