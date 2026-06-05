import os

# 定义文件名
file_names = ['nwpu_tag_vocab_512.txt', 'rsicd_tag_vocab_512.txt', 'rsitmd_tag_vocab_512.txt']
base_dir = os.path.dirname(os.path.abspath(__file__))

# 初始化一个空集合来存储所有单词
unique_words = set()

# 遍历每个文件
for file_name in file_names:
    # 打开文件并读取每一行
    with open(os.path.join(base_dir, file_name), 'r', encoding='utf-8') as file:
        # 将每一行的单词添加到集合中（使用split分割单词）
        lines = file.readlines()[:512]
                # 将每一行的单词添加到集合中（使用split分割单词）
        for line in lines:
            unique_words.update(line.split())

        # unique_words.update(file.read().split())

# 将唯一的单词写入新文件（可以根据需要修改文件名）
output_file_name = os.path.join(base_dir, 'merged_words.txt')
with open(output_file_name, 'w', encoding='utf-8') as output_file:
    # 将集合中的单词写入文件
    output_file.write('\n'.join(unique_words))

print(f"合并完成，并将唯一单词写入文件：{output_file_name}")
