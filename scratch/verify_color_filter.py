import sys
import os
import re

# nodesディレクトリの親ディレクトリをsys.pathに追加してインポート可能にする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes.color_filter.color_filter import ColorFilter

def run_tests():
    filter_node = ColorFilter()
    
    print("--- 動作検証テスト開始 ---")
    
    # 1. 既存のハードコードされている除外ワード（monochromeなど）が機能することを確認
    text = "A monochrome photo of a black and white cat, grayscale representation."
    res, = filter_node.filter_text(text)
    print("テスト 1 (ハードコードされたワードの除外):", res)
    assert "monochrome" not in res.lower()
    assert "black and white" not in res.lower()
    assert "grayscale" not in res.lower()
    assert res == "A photo of a cat, representation."
    
    # 2. カンマで区切られた任意のカスタムワード（例: blue, eyes）のテスト
    text = "A cute blue bird with bright eyes."
    res, = filter_node.filter_text(text, exclude_words="blue, eyes")
    print("テスト 2 (カンマ区切りのカスタム除外):", res)
    assert "blue" not in res.lower()
    assert "eyes" not in res.lower()
    assert "bird" in res.lower()
    
    # 3. 改行で区切られた任意のカスタムワード（例: pink, wings）のテスト
    text = "A cute pink bird with small wings."
    res, = filter_node.filter_text(text, exclude_words="pink\nwings")
    print("テスト 3 (改行区切りのカスタム除外):", res)
    assert "pink" not in res.lower()
    assert "wings" not in res.lower()
    assert "bird" in res.lower()
    
    # 4. 英単語の境界判定テスト
    # "black"を除外ワードに指定した場合、単語「black」は消えるが、「blackberry」内の「black」は残るべき
    text = "I have a black cat eating a blackberry."
    res, = filter_node.filter_text(text, exclude_words="black")
    print("テスト 4 (英単語の境界判定):", res)
    assert "black cat" not in res.lower()
    assert "blackberry" in res.lower()
    assert res == "I have a cat eating a blackberry."
    
    # 5. 日本語（漢字・ひらがな・カタカナ）の除外指定テスト
    text = "これは白黒の猫のモノクロ画像です。"
    res, = filter_node.filter_text(text, exclude_words="画像")
    print("テスト 5 (日本語カスタム除外):", res)
    assert "画像" not in res
    # 「白黒」と「モノクロ」は元のハードコードリストによって消去されるはず
    assert "白黒" not in res
    assert "モノクロ" not in res
    assert res == "これはの猫のです。"
    
    # 6. 記号を含む文字列のエスケープテスト
    text = "This is a B&W(monochrome) picture."
    res, = filter_node.filter_text(text, exclude_words="B&W(monochrome)")
    print("テスト 6 (記号を含む除外のエスケープ):", res)
    assert "B&W(monochrome)" not in res
    assert res == "This is a picture."
    
    # 7. 後方互換性テスト（引数1つで呼び出された場合）
    text = "A monochrome photo."
    res, = filter_node.filter_text(text)
    print("テスト 7 (後方互換性 - 引数なし):", res)
    assert "monochrome" not in res.lower()
    assert res == "A photo."

    print("\n--- すべてのテストが正常に通過しました！ ---")

if __name__ == "__main__":
    run_tests()
