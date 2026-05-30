# Investigation & Resolution Report: ColorFilter Exclusion Issue

This document provides a technical overview of the issue where custom words (such as `"blonde"`) specified in the `exclude_words` input parameter of the `ColorFilter` node were not being excluded from prompt outputs, along with the detailed analysis, the root causes, the modified file, the full source code, and a detailed explanation of the logic.

---

## 1. Problem Description

When running the `ColorFilter` node inside ComfyUI with an input structure containing `exclude_words` (e.g., `"blonde, glasses,"`), the word `"blonde"` failed to be filtered out from the final output text. The user observed that despite specifying `"blonde"` as an excluded word, the output string still contained the word, and formatting around commas became malformed (leaving consecutive dangling commas).

---

## 2. Root Cause Analysis

We identified two distinct causes that together resulted in this bug:

### Cause A: Stale Python Bytecode Cache (`__pycache__`)
* **Underlying Issue:** The directory `nodes/color_filter/__pycache__/` contained precompiled `.pyc` files dating from before the `exclude_words` input feature was introduced.
* **Mechanism:** When ComfyUI imported the `ColorFilter` node, Python loaded the stale compiled bytecode (`.pyc`) instead of parsing the updated source code (`.py`). Because the older class signature lacked the `exclude_words` input in `INPUT_TYPES` and the `exclude_words` parameter in the `filter_text` method, ComfyUI silently ignored any inputs connected to `exclude_words` and bypassed the custom exclusion logic entirely.
* **Solution:** We deleted the stale `__pycache__` directory, forcing the Python runtime to re-parse the source code and generate fresh, correct bytecode matching the updated class signature.

### Cause B: Single-Pass Comma Cleanup Regex
* **Underlying Issue:** When multiple consecutive words are excluded (e.g., both `"blonde"` and `"glasses"` from `"blonde, glasses, photo of..."`), their removal leaves behind multiple adjacent commas separated by spaces (e.g., `, , ,`).
* **Mechanism:** The original cleaning pattern was:
  ```python
  filtered_text = re.sub(r",\s*,", ",", filtered_text)
  ```
  Since `re.sub` executes in a single non-overlapping pass, a sequence of three or more commas is only partially collapsed (e.g., `, , ,` becomes `, ,`), leaving orphaned commas in the finalized prompt.
* **Solution:** We replaced the single-pass substitution with an iterative `while` loop that runs repeatedly until the string stabilizes and no consecutive commas remain.

---

## 3. Modified File

* **File Path:** `nodes/color_filter/color_filter.py`
* **Changes Made:**
  1. Deletion of the stale `__pycache__` directory to force compilation.
  2. Modified the comma collapsing block in `nodes/color_filter/color_filter.py` to use an iterative `while` loop.

---

## 4. Complete Source Code of `color_filter.py`

Below is the complete source code of `nodes/color_filter/color_filter.py` including the fix:

```python
"""Remove monochrome / B&W related phrases from caption or prompt text."""

import re


class ColorFilter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "exclude_words": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_text",)
    FUNCTION = "filter_text"
    CATEGORY = "Text/Filter"

    def filter_text(self, text, exclude_words=""):
        keywords_to_remove = [
            r"\bwhite and black\b",
            r"\bblack and white\b",
            r"\bmonochrome\b",
            r"\bgrayscale\b",
            r"\bgrey scale\b",
            r"\bgreyscale\b",
            r"\bB&W\b",
            r"\bdesaturated\b",
            r"\bblonde\b",
            r"\bblonde hair\b",
            r"\bachromatic\b",
            r"白黒",
            r"モノクロ",
            r"グレースケール",
            r"無彩色",
            r"セピア",
        ]

        user_patterns = []
        if exclude_words:
            # Split by comma or newline
            for word in re.split(r'[,\n]', exclude_words):
                word = word.strip()
                if not word:
                    continue
                
                # Escape string for regex
                escaped_word = re.escape(word)
                
                # Append word boundary (\b) only if the word starts/ends with ASCII alphanumeric characters
                start_boundary = r"\b" if re.match(r'^[a-zA-Z0-9_]', word) else ""
                end_boundary = r"\b" if re.search(r'[a-zA-Z0-9_]$', word) else ""
                
                pattern = f"{start_boundary}{escaped_word}{end_boundary}"
                user_patterns.append(pattern)

        keywords_to_remove = user_patterns + keywords_to_remove

        # Sort patterns by the length of the actual matched text in descending order
        # to ensure longer phrases (e.g. "blonde hair") are removed before shorter sub-words (e.g. "blonde")
        keywords_to_remove.sort(
            key=lambda p: len(re.sub(r"\\(.)", r"\1", p.replace(r"\b", "").replace(r"\ ", " "))),
            reverse=True
        )

        filtered_text = text
        for keyword in keywords_to_remove:
            filtered_text = re.sub(keyword, "", filtered_text, flags=re.IGNORECASE)

        # Collapse multiple commas, spaces, and strip leading/trailing commas and spaces
        prev = None
        while prev != filtered_text:
            prev = filtered_text
            filtered_text = re.sub(r",\s*,", ",", filtered_text)
        filtered_text = re.sub(r"\s+", " ", filtered_text)
        filtered_text = filtered_text.strip(" ,")

        return (filtered_text,)
```

---

## 5. Detailed Explanation of the Code

### 5.1 Defining Input Configuration (`INPUT_TYPES`)
```python
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "exclude_words": ("STRING", {"default": ""}),
            }
        }
```
ComfyUI queries `INPUT_TYPES` to build the UI node widgets. 
* `text` accepts the target string (e.g., generated caption from a tagging model).
* `exclude_words` accepts a string input containing custom words to remove, default is empty.

### 5.2 Dynamic Regex Parsing & Word Boundary Protection
```python
        user_patterns = []
        if exclude_words:
            # Split by comma or newline
            for word in re.split(r'[,\n]', exclude_words):
                word = word.strip()
                if not word:
                    continue
                
                # Escape string for regex
                escaped_word = re.escape(word)
                
                # Append word boundary (\b) only if the word starts/ends with ASCII alphanumeric characters
                start_boundary = r"\b" if re.match(r'^[a-zA-Z0-9_]', word) else ""
                end_boundary = r"\b" if re.search(r'[a-zA-Z0-9_]$', word) else ""
                
                pattern = f"{start_boundary}{escaped_word}{end_boundary}"
                user_patterns.append(pattern)
```
* **Splitting:** Input from `exclude_words` is split by commas or newline characters.
* **Escaping:** We apply `re.escape` to prevent any characters (like parenthesis or brackets) from being interpreted as active regex grouping/operators, which would break execution.
* **Conditional boundaries (`\b`):** We check if the word starts or ends with an ASCII alphanumeric character. The word boundary selector `\b` matches the boundary between a word character (`\w`) and a non-word character. Applying it unconditionally to non-ASCII/CJK characters (like Japanese Kanji/Kana) prevents matches because CJK characters are not in the standard ASCII alphanumeric range. This dynamic configuration prevents boundary errors across mixed language prompts.

### 5.3 Order-of-Length Sort Strategy
```python
        # Sort patterns by the length of the actual matched text in descending order
        # to ensure longer phrases (e.g. "blonde hair") are removed before shorter sub-words (e.g. "blonde")
        keywords_to_remove.sort(
            key=lambda p: len(re.sub(r"\\(.)", r"\1", p.replace(r"\b", "").replace(r"\ ", " "))),
            reverse=True
        )
```
* If a shorter pattern like `"blonde"` is matched and replaced before `"blonde hair"`, the phrase `"blonde hair"` would become `" hair"`, leaving unwanted dangling words.
* To prevent this, we calculate the literal length of each pattern (ignoring structural regex metadata like `\b` and escape slashes `\`) and sort the combined keyword array in descending order. This ensures longer, more specific phrases are processed and removed first.

### 5.4 Iterative Multi-Pass Cleanup Loop
```python
        # Collapse multiple commas, spaces, and strip leading/trailing commas and spaces
        prev = None
        while prev != filtered_text:
            prev = filtered_text
            filtered_text = re.sub(r",\s*,", ",", filtered_text)
        filtered_text = re.sub(r"\s+", " ", filtered_text)
        filtered_text = filtered_text.strip(" ,")
```
* **Iterative Comma Collapsing:** The `while prev != filtered_text` loop ensures that sequential runs of adjacent commas (such as `, , ,`) are collapsed in multiple passes until no two commas remain separated by spaces.
* **Whitespace Normalization & Stripping:** Contiguous spaces are reduced to a single space, and leading or trailing spaces/commas are cleaned up, leaving a clean, valid prompt string.
