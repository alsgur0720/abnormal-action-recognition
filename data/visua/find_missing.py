import matplotlib.pyplot as plt
import numpy as np

f = open("/home/vimlab/workspace/source/visualization/my_visualization/NTU_RGBD_samples_with_missing_skeletons.txt", 'r')
#S001C002P005R002A008

# 클래스 저장
class_list = []
while True:
    line = f.readline()
    if not line:
        break
    class_num = line[-4:-1]
    #print(type(class_num))
    class_list.append(class_num)
f.close()

# 중복 제거
missing_class = set(class_list)
# print(len(missing_class)) # 51


# 클래스 딕셔너리 key 값은 '0xx' 형태
class_dict = {}
for i in range(1, 61):
    a = format(i, '03')
    class_dict[a]=0

# 딕셔너리에 값 채우기
for cls in class_list:
    class_dict[cls]+=1


dict_values = class_dict.values() 
dict_values = class_dict.values() 
#print(dict_values)

keys_np = np.arange(1, 61)
values_np = np.array(list(dict_values))
# print(values_np)
# print(keys_np)

# 그래프 설정
bar = plt.bar(np.arange(60), values_np)
plt.xticks(np.arange(60), keys_np)
plt.xlabel('Class label')
plt.ylabel('Missing skeleton number')

# 막대 그래프 위에 수치 기입
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % height, ha='center', va='bottom', size=10)
plt.show()


