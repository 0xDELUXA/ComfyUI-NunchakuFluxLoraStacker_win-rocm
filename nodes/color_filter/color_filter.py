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
            # カンマまたは改行で分割
            for word in re.split(r'[,\n]', exclude_words):
                word = word.strip()
                if not word:
                    continue
                
                # 正規表現のエスケープ処理
                escaped_word = re.escape(word)
                
                # ASCII英数字で始まる/終わる場合のみ単語の境界(\b)を付与する
                start_boundary = r"\b" if re.match(r'^[a-zA-Z0-9_]', word) else ""
                end_boundary = r"\b" if re.search(r'[a-zA-Z0-9_]$', word) else ""
                
                pattern = f"{start_boundary}{escaped_word}{end_boundary}"
                user_patterns.append(pattern)

        keywords_to_remove = user_patterns + keywords_to_remove

        filtered_text = text
        for keyword in keywords_to_remove:
            filtered_text = re.sub(keyword, "", filtered_text, flags=re.IGNORECASE)

        filtered_text = re.sub(r"\s+", " ", filtered_text).strip()

        return (filtered_text,)
