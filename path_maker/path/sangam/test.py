# 파일에서 데이터를 읽어옴
with open('sangam_avoid_path_3.txt', 'r') as file:
    lines = file.readlines()

# 중복된 줄의 인덱스를 저장할 딕셔너리
duplicate_indices = {}

# 각 줄에 대해 인덱스를 확인하고 딕셔너리에 저장
for i, line in enumerate(lines):
    if line in duplicate_indices:
        duplicate_indices[line].append(i)
    else:
        duplicate_indices[line] = [i]

# 중복된 줄이 있는지 확인하고 출력
for line, indices in duplicate_indices.items():
    if len(indices) > 1:
        print(f"중복된 라인 '{line.strip()}'의 인덱스: {indices}")
