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
