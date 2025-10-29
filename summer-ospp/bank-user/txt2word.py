import os
import time
from docx import Document
from threading import Thread

def process_file(txt_path, word_path):
    try:
        print(f"开始处理文件: {txt_path}")
        start_time = time.time()
        
        # 读取文件（超时设置）
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print(f"读取完成，长度: {len(content)} 字符")
        
        # 创建 Word 文档
        doc = Document()
        for line in content.split('\n'):
            doc.add_paragraph(line)
        
        doc.save(word_path)
        print(f"保存成功: {word_path} (耗时: {time.time() - start_time:.2f}s)")
    
    except Exception as e:
        print(f"处理失败: {e}")

def txt_to_word(source_folder):
    if not os.path.exists(source_folder):
        print(f"文件夹不存在: {source_folder}")
        return
    
    for i in range(10, 11):  # 只处理 10.txt（测试用）
        txt_filename = f"{i}.txt"
        word_filename = f"{i}.docx"
        txt_path = os.path.join(source_folder, txt_filename)
        word_path = os.path.join(source_folder, word_filename)
        
        if not os.path.exists(txt_path):
            print(f"文件不存在: {txt_filename}")
            continue
        
        # 启动线程（避免主程序卡死）
        t = Thread(target=process_file, args=(txt_path, word_path))
        t.start()
        t.join(timeout=30)  # 设置 30 秒超时
        
        if t.is_alive():
            print(f"处理超时: {txt_filename}")

if __name__ == "__main__":
    folder_path = "E:/guolei/Documents/translate"  # 替换为你的路径
    txt_to_word(folder_path)
    print("处理结束")