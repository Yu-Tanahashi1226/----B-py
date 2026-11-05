"""Adapters for the uploaded two-stage B/S/O layout and inning/score OCR."""
import math
import re


SLOTS = {
    'ball': [(0.185, 0.835), (0.235, 0.835), (0.285, 0.835)],
    'strike': [(0.500, 0.835), (0.555, 0.835)],
    'out': [(0.835, 0.835), (0.895, 0.835)],
}
BASES = {'first': (0.236, 0.356), 'second': (0.173, 0.216), 'third': (0.110, 0.356)}
THRESHOLDS = {'ball': 0.25, 'strike': 0.35, 'out': 0.30, 'runner': 0.30}


def board_layout_status(box, width, height, profile=None):
    """The field coordinates apply only to the configured corner-board layout."""
    if box is None:
        return 'no_detection'
    if not profile:
        return 'not_checked'
    x0, y0, x1, y1 = box
    bw, bh = x1-x0, y1-y0
    valid = (bh > 0 and x0/width >= profile['minimum_x'] and y0/height >= profile['minimum_y']
             and profile['width_range'][0] <= bw/width <= profile['width_range'][1]
             and profile['aspect_range'][0] <= bw/bh <= profile['aspect_range'][1])
    return 'supported_corner_layout' if valid else 'unsupported_layout'


def inner_state(detections, width, height, distance=0.08):
    accepted = {name: {} for name in SLOTS}
    runners = set()
    for item in detections:
        label = item['label'].strip().casefold()
        if label not in THRESHOLDS or item['confidence'] < THRESHOLDS[label]:
            continue
        x0, y0, x1, y1 = item['box']
        nx, ny = (x0 + x1) / (2 * width), (y0 + y1) / (2 * height)
        choices = dict(enumerate(SLOTS[label])) if label in SLOTS else BASES
        key, reference = min(choices.items(), key=lambda pair: math.hypot(nx - pair[1][0], ny - pair[1][1]))
        if math.hypot(nx - reference[0], ny - reference[1]) > distance:
            continue
        if label == 'runner':
            runners.add(key)
        else:
            current = accepted[label].get(key)
            if current is None or item['confidence'] > current['confidence']:
                accepted[label][key] = item
    count = dict(balls=len(accepted['ball']), strikes=len(accepted['strike']), outs=len(accepted['out']))
    # An empty inner detection cannot distinguish 0-0-0 from a failed detection.
    if sum(count.values()) == 0:
        return None, None, 'no_accepted_bso_lamps_unknown'
    runner_state = {name: name in runners for name in BASES}
    return count, runner_state, 'two_stage_yolo_slots'


def preprocess_region(image, yellow=False, scale=4):
    import cv2
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if yellow:
        image = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (15, 80, 80), (45, 255, 255))
    else:
        _, image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
    image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def parse_score(text):
    normalized = text.strip().translate(str.maketrans({'O': '0', 'o': '0', 'I': '1', 'l': '1', '|': '1'}))
    if re.fullmatch(r'\d{1,2}', normalized):
        return int(normalized)
    return None


def parse_inning(text, max_inning=20):
    normalized = text.replace(' ', '').translate(str.maketrans({'O': '0', 'o': '0', 'I': '1', 'l': '1', '|': '1'}))
    matches = re.findall(r'\d+', normalized)
    inning = int(matches[0]) if len(matches) == 1 and 1 <= int(matches[0]) <= max_inning else None
    bottom = 'ウラ' in normalized or '裏' in normalized
    top = 'オモテ' in normalized or '表' in normalized
    half = 'bottom' if bottom and not top else 'top' if top and not bottom else None
    return inning, half


def field_text(tokens, region_height, minimum_confidence=0.6, minimum_height_ratio=0.25):
    """Ignore tiny OCR noise and order single-line field tokens by horizontal position."""
    accepted = []
    for token in tokens:
        polygon = token.get('polygon') or []
        if token.get('confidence', 0) < minimum_confidence or not polygon:
            continue
        height = max(p[1] for p in polygon) - min(p[1] for p in polygon)
        if height < region_height * minimum_height_ratio:
            continue
        accepted.append(token)
    accepted.sort(key=lambda t: sum(p[0] for p in t['polygon']) / len(t['polygon']))
    return ''.join(t['text'] for t in accepted), accepted
