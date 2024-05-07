import json

def extract_points(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as f_out:
        for i in range(len(data)):
            for point in data[i]["points"]:
                # Joining the two values in the point array with a tab separator
                line = '\t'.join(map(str, point[:2])) + '\n'
                f_out.write(line)

# 입력 파일과 출력 파일 경로 설정
input_json_file = 'input_2.json'
output_txt_file = 'output_2.txt'

# 함수 호출하여 작업 실행
extract_points(input_json_file, output_txt_file)
