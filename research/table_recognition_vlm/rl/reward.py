"""Deterministic, model-free rewards for table HTML generation."""

from dataclasses import dataclass
from functools import lru_cache
from html import escape
from html.parser import HTMLParser
import re


STRUCTURE_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


@dataclass(frozen=True)
class ParsedTable:
    valid: bool
    canonical: tuple[str, ...]
    structure: tuple[str, ...]
    content: tuple[str, ...]
    num_rows: int
    num_cols: int
    num_cells: int
    has_merged_cells: bool


def _positive_int(value, default=1):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default, False
    return (parsed, parsed > 0) if parsed > 0 else (default, False)


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.inside_table = False
        self.completed_tables = 0
        self.invalid = False
        self.stack = []
        self.canonical = []
        self.structure = []
        self.rows = []
        self.current_row = None
        self.cells = []
        self.current_cell = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = [(name.lower(), value or "") for name, value in attrs]
        if not self.inside_table:
            if tag != "table" or self.completed_tables:
                self.invalid = True
                return
            self.inside_table = True

        if tag == "table" and self.stack:
            self.invalid = True

        normalized_attrs = tuple(sorted((name, " ".join(value.split())) for name, value in attrs))
        rendered_attrs = "".join(
            f' {name}="{escape(value, quote=True)}"' for name, value in normalized_attrs
        )
        self.canonical.append(f"<{tag}{rendered_attrs}>")

        if tag in STRUCTURE_TAGS:
            if tag in {"td", "th"}:
                attr_map = dict(normalized_attrs)
                rowspan, rowspan_valid = _positive_int(attr_map.get("rowspan"))
                colspan, colspan_valid = _positive_int(attr_map.get("colspan"))
                self.invalid |= not rowspan_valid and "rowspan" in attr_map
                self.invalid |= not colspan_valid and "colspan" in attr_map
                self.structure.append(f"<{tag}:r{rowspan}:c{colspan}>")
                if self.current_row is None or self.current_cell is not None:
                    self.invalid = True
                cell = {"tag": tag, "rowspan": rowspan, "colspan": colspan, "text": []}
                self.cells.append(cell)
                if self.current_row is not None:
                    self.current_row.append(cell)
                self.current_cell = cell
            else:
                self.structure.append(f"<{tag}>")
                if tag == "tr":
                    if self.current_row is not None:
                        self.invalid = True
                    self.current_row = []
                    self.rows.append(self.current_row)

        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag.lower() not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if not self.inside_table:
            self.invalid = True
            return
        if tag in VOID_TAGS:
            self.invalid = True
            return
        if not self.stack or self.stack[-1] != tag:
            self.invalid = True
            return

        self.stack.pop()
        self.canonical.append(f"</{tag}>")
        if tag in STRUCTURE_TAGS:
            if tag in {"td", "th"}:
                self.structure.append(f"</{tag}>")
                self.current_cell = None
            else:
                self.structure.append(f"</{tag}>")
                if tag == "tr":
                    self.current_row = None
        if tag == "table":
            self.inside_table = False
            self.completed_tables += 1

    def handle_data(self, data):
        normalized = " ".join(data.split())
        if not self.inside_table:
            if normalized:
                self.invalid = True
            return
        if normalized:
            self.canonical.append(escape(normalized, quote=False))
            if self.current_cell is not None:
                self.current_cell["text"].append(normalized)

    def handle_comment(self, data):
        if data.strip():
            self.invalid = True

    def handle_decl(self, decl):
        self.invalid = True

    def handle_pi(self, data):
        self.invalid = True

    def unknown_decl(self, data):
        self.invalid = True

    def error(self, message):
        self.invalid = True


def _logical_shape(rows):
    occupied_until = {}
    width = 0
    height = len(rows)
    for row_index, row in enumerate(rows):
        occupied_until = {
            column: end_row for column, end_row in occupied_until.items() if end_row > row_index
        }
        column = 0
        for cell in row:
            colspan = cell["colspan"]
            while any(occupied_until.get(candidate, 0) > row_index for candidate in range(column, column + colspan)):
                column += 1
            end_column = column + colspan
            end_row = row_index + cell["rowspan"]
            for candidate in range(column, end_column):
                occupied_until[candidate] = max(occupied_until.get(candidate, 0), end_row)
            width = max(width, end_column)
            height = max(height, end_row)
            column = end_column
        if occupied_until:
            width = max(width, max(occupied_until) + 1)
    return height, width


@lru_cache(maxsize=512)
def parse_table_html(value):
    if not isinstance(value, str):
        return ParsedTable(False, (), (), (), 0, 0, 0, False)

    parser = _TableParser()
    try:
        parser.feed(value)
        parser.close()
    except (TypeError, ValueError):
        parser.invalid = True

    content = []
    for cell in parser.cells:
        content.extend(re.findall(r"\S+", " ".join(cell["text"])))
        content.append("<CELL>")
    valid = (
        not parser.invalid
        and parser.completed_tables == 1
        and not parser.inside_table
        and not parser.stack
        and bool(parser.rows)
        and bool(parser.cells)
    )
    num_rows, num_cols = _logical_shape(parser.rows)
    return ParsedTable(
        valid=valid,
        canonical=tuple(parser.canonical),
        structure=tuple(parser.structure),
        content=tuple(content),
        num_rows=num_rows,
        num_cols=num_cols,
        num_cells=len(parser.cells),
        has_merged_cells=any(
            cell["rowspan"] > 1 or cell["colspan"] > 1 for cell in parser.cells
        ),
    )


def extract_table_html(value):
    """Extract one complete table from dataset labels that may include an HTML document."""
    if not isinstance(value, str):
        raise TypeError(f"table_html must be a string, got {type(value).__name__}")
    lowercase = value.lower()
    start = lowercase.find("<table")
    end_tag = "</table>"
    end = lowercase.rfind(end_tag)
    if start < 0 or end < start:
        raise ValueError("table_html does not contain a complete <table>...</table> element")
    return value[start : end + len(end_tag)].strip()


@lru_cache(maxsize=512)
def _reference_table(value):
    try:
        return parse_table_html(extract_table_html(value))
    except (TypeError, ValueError):
        return ParsedTable(False, (), (), (), 0, 0, 0, False)


def completion_text(completion):
    """Extract assistant text from TRL's standard or conversational completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion_text(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        parts = []
        for item in completion:
            if isinstance(item, dict) and item.get("type") not in (None, "text"):
                continue
            parts.append(completion_text(item))
        return "".join(parts)
    return ""


def _aligned_similarity(prediction, reference):
    total = max(len(prediction), len(reference))
    if total == 0:
        return 1.0
    return sum(left == right for left, right in zip(prediction, reference)) / total


def format_reward(completions, **_):
    return [float(parse_table_html(completion_text(item)).valid) for item in completions]


def exact_reward(completions, table_html, **_):
    rewards = []
    for completion, reference in zip(completions, table_html):
        prediction = parse_table_html(completion_text(completion))
        target = _reference_table(reference)
        rewards.append(float(prediction.valid and prediction.canonical == target.canonical))
    return rewards


def structure_reward(completions, table_html, **_):
    rewards = []
    for completion, reference in zip(completions, table_html):
        prediction = parse_table_html(completion_text(completion))
        target = _reference_table(reference)
        rewards.append(
            _aligned_similarity(prediction.structure, target.structure)
            if prediction.valid and target.valid
            else 0.0
        )
    return rewards


def content_reward(completions, table_html, **_):
    rewards = []
    for completion, reference in zip(completions, table_html):
        prediction = parse_table_html(completion_text(completion))
        target = _reference_table(reference)
        rewards.append(
            _aligned_similarity(prediction.content, target.content)
            if prediction.valid and target.valid
            else 0.0
        )
    return rewards


def reasoning_metadata_reward(
    completions,
    num_rows,
    num_cols,
    num_cells,
    has_merged_cells,
    validation_passed=None,
    **_,
):
    """Check predictions against compact facts materialized from the reasoning trace."""
    rewards = []
    for index, completion in enumerate(completions):
        prediction = parse_table_html(completion_text(completion))
        if not prediction.valid:
            rewards.append(0.0)
            continue
        checks = (
            prediction.num_rows == int(num_rows[index]),
            prediction.num_cols == int(num_cols[index]),
            prediction.num_cells == int(num_cells[index]),
            prediction.has_merged_cells == bool(has_merged_cells[index]),
        )
        rewards.append(sum(checks) / len(checks))
    return rewards


REWARD_FUNCTIONS = [
    format_reward,
    exact_reward,
    structure_reward,
    content_reward,
    reasoning_metadata_reward,
]
REWARD_WEIGHTS = [0.25, 1.0, 1.0, 1.0, 0.5]
